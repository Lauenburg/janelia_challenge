import torch


class CellMapLossWrapper(torch.nn.modules.loss._Loss):
    """
    Wrapper for any PyTorch loss function that is applied to the output of a model and the target.

    Because the target can contain NaN values, the loss function is applied only to the non-NaN values.
    This is done by multiplying the loss by a mask that is 1 where the target is not NaN and 0 where the target is NaN.
    The loss is then averaged across the non-NaN values.

    Parameters
    ----------
    loss_fn : torch.nn.modules.loss._Loss or torch.nn.modules.loss._WeightedLoss
        The loss function to apply to the output and target.
    **kwargs
        Keyword arguments to pass to the loss function.
    """

    def __init__(
        self,
        loss_fn: torch.nn.modules.loss._Loss | torch.nn.modules.loss._WeightedLoss,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        self.kwargs["reduction"] = "none"

        try:
            self.loss_fn = loss_fn(**self.kwargs)
        except TypeError as e:
            if "reduction" in str(e).lower() or "unexpected keyword" in str(e).lower():
                kwargs_no_reduction = {k: v for k, v in self.kwargs.items() if k != "reduction"}
                self.loss_fn = loss_fn(**kwargs_no_reduction)
            else:
                raise

        self._supports_class_index = hasattr(self.loss_fn, "set_class_index")
        self._last_unclamped_loss: float = 0.0

    def calc_loss(
        self,
        outputs: torch.Tensor,
        target: torch.Tensor,
        class_index: int | None = None,
    ) -> torch.Tensor:
        target = target.to(dtype=outputs.dtype)
        mask = target.isnan().logical_not()
        target_for_loss = target.nan_to_num(0.0).clamp(0.0, 1.0)

        loss_fn_to_use = self.loss_fn
        if self._supports_class_index and class_index is not None:
            loss_fn_to_use.set_class_index(class_index)

        loss = loss_fn_to_use(outputs, target_for_loss)
        masked_loss = loss * mask

        mask_sum = mask.sum()
        if mask_sum > 0:
            loss_mean = masked_loss.sum() / mask_sum.clamp(min=1)

            # More robust negative loss handling
            if loss_mean < -1e-6:
                print(f"WARNING: Negative loss {loss_mean.item():.6e}. Skipping batch.")
                return torch.tensor(float("nan"), device=outputs.device, dtype=outputs.dtype)
            elif loss_mean < 0:
                loss_mean = torch.clamp(loss_mean, min=0.0)

            # Handle NaN/Inf more robustly
            if not torch.isfinite(loss_mean):
                finite_mask = torch.isfinite(masked_loss) & mask
                if finite_mask.sum() > 0:
                    loss_mean = masked_loss[finite_mask].sum() / finite_mask.sum()
                else:
                    return torch.tensor(float("nan"), device=outputs.device, dtype=outputs.dtype)

            self._last_unclamped_loss = float(loss_mean.item())
            return loss_mean

        return torch.tensor(float("nan"), device=outputs.device, dtype=outputs.dtype)

    def forward(
        self,
        outputs: dict | torch.Tensor,
        targets: dict | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(targets, dict):
            valid_class_losses: list[torch.Tensor] = []

            if isinstance(outputs, dict):
                for class_idx, (key, target) in enumerate(targets.items()):
                    class_loss = self.calc_loss(outputs[key], target, class_index=class_idx)
                    if torch.isfinite(class_loss):
                        valid_class_losses.append(class_loss)
            else:
                for class_idx, target in enumerate(targets.values()):
                    class_loss = self.calc_loss(outputs[class_idx], target, class_index=class_idx)
                    if torch.isfinite(class_loss):
                        valid_class_losses.append(class_loss)

            if len(valid_class_losses) == 0:
                first_target = next(iter(targets.values()))
                return torch.tensor(float("nan"), device=first_target.device, dtype=torch.float32)

            loss = sum(valid_class_losses) / len(valid_class_losses)
        else:
            loss = self.calc_loss(outputs, targets)

        return loss