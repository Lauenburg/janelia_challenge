## Unreleased

- Training config tweaks (`examples/train_2D.py`): switch to AdamW + cosine LR, grad clip 0.5, grad accumulation 2, balanced Dice/BCE (0.6/0.4, smooth 1e-3), skip-all-zero/force-classes, lighter blur-only intensity aug, 150 epochs with validation caps.
- UNet 2D (`models/unet_model_2D.py`): replace BatchNorm with GroupNorm (8 groups); always use bilinear upsampling to avoid checkerboard artifacts; trilinear flag kept for API compatibility.
- Inference (`predict.py`): refactored 2.5D orthoplane path, added env toggles (`CSC_PRED_DISABLE_ORTHO`, `CSC_PRED_MAX_INDICES`), per-sample aggregation, clearer logging, safer singleton handling and device placement.
- Dataloading/splits: only build validation loader when datasets exist; enforce at least one train/val split when possible, warn on single-dataset cases.
- Loss wrapper (`utils/loss.py`): more robust to NaN/Inf/negative losses, supports class-index-aware losses, ignores invalid class losses instead of crashing.