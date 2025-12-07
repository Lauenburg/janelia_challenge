import os
import tempfile
from glob import glob
from typing import Any

import numpy as np
import torch
import torchvision.transforms.v2 as T
from cellmap_data import CellMapDatasetWriter, CellMapImage
from cellmap_data.utils import (
    array_has_singleton_dim,
    is_array_2D,
    permute_singleton_dimension,
)
from cellmap_data.transforms.augment import NaNtoNum, Normalize
from tqdm import tqdm
from upath import UPath

from .config import CROP_NAME, PREDICTIONS_PATH, RAW_NAME, SEARCH_PATH
from .models import get_model
from .utils import load_safe_config, get_test_crops
from .utils.datasplit import get_formatted_fields, get_raw_path


def predict_orthoplanes(
    model: torch.nn.Module, dataset_writer_kwargs: dict[str, Any], batch_size: int
):
    print("Predicting orthogonal planes.")
    tmp_dir = tempfile.TemporaryDirectory()
    print(f"Temporary directory for predictions: {tmp_dir.name}")
    
    for axis in range(3):
        temp_kwargs = dataset_writer_kwargs.copy()
        temp_kwargs["target_path"] = os.path.join(
            tmp_dir.name, "output.zarr", str(axis)
        )
        input_arrays = {k: v.copy() for k, v in temp_kwargs["input_arrays"].items()}
        target_arrays = {k: v.copy() for k, v in temp_kwargs["target_arrays"].items()}
        permute_singleton_dimension(input_arrays, axis)
        permute_singleton_dimension(target_arrays, axis)
        temp_kwargs["input_arrays"] = input_arrays
        temp_kwargs["target_arrays"] = target_arrays
        _predict(model, temp_kwargs, batch_size=batch_size)

    dataset_writer = CellMapDatasetWriter(**dataset_writer_kwargs)
    single_axis_images = {
        array_name: {
            label: [
                CellMapImage(
                    os.path.join(tmp_dir.name, "output.zarr", str(axis), label),
                    target_class=label,
                    target_scale=array_info["scale"],
                    target_voxel_shape=array_info["shape"],
                    pad=True,
                    pad_value=0,
                )
                for axis in range(3)
            ]
            for label in dataset_writer_kwargs["classes"]
        }
        for array_name, array_info in dataset_writer_kwargs["target_arrays"].items()
    }

    print("Combining predictions.")
    for batch in tqdm(dataset_writer.loader(batch_size=batch_size), dynamic_ncols=True):
        for idx in batch["idx"]:
            idx_val = int(idx) if hasattr(idx, "item") else int(idx)
            sample_out = {}
            for array_name, images in single_axis_images.items():
                sample_out[array_name] = {}
                for label in dataset_writer_kwargs["classes"]:
                    average_prediction = []
                    for image in images[label]:
                        average_prediction.append(image[dataset_writer.get_center(idx)])
                    average_prediction = torch.stack(average_prediction).mean(dim=0)
                    sample_out[array_name][label] = average_prediction
            dataset_writer[idx_val] = sample_out

    tmp_dir.cleanup()


def _predict(
    model: torch.nn.Module, dataset_writer_kwargs: dict[str, Any], batch_size: int
):
    value_transforms = T.Compose(
        [
            Normalize(),
            T.ToDtype(torch.float, scale=True),
            NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
        ],
    )

    dataset_writer = CellMapDatasetWriter(
        **dataset_writer_kwargs, raw_value_transforms=value_transforms
    )
    dataloader = dataset_writer.loader(batch_size=batch_size)
    model.eval()
    
    singleton_dim = np.where(
        [s == 1 for s in dataset_writer_kwargs["input_arrays"]["input"]["shape"]]
    )[0]
    singleton_dim = singleton_dim[0] if singleton_dim.size > 0 else None
    classes = dataset_writer_kwargs["classes"]
    
    max_indices = int(os.getenv("CSC_PRED_MAX_INDICES", "0"))
    processed = 0
    
    device = dataset_writer_kwargs.get("device", "cpu")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, dynamic_ncols=True):
            if max_indices > 0 and processed >= max_indices:
                break
                
            inputs = batch["input"].to(device)
            if singleton_dim is not None:
                inputs = inputs.squeeze(dim=singleton_dim + 2)
            
            outputs = model(inputs)
            
            if singleton_dim is not None:
                outputs = outputs.unsqueeze(dim=singleton_dim + 2)
            
            for b, idx in enumerate(batch["idx"]):
                if max_indices > 0 and processed >= max_indices:
                    break
                    
                idx_val = int(idx) if hasattr(idx, "item") else int(idx)
                sample_out = {"output": {}}
                
                for c, label in enumerate(classes):
                    pred = outputs[b, c, ...]
                    if pred.dim() == 2:
                        pred = pred.unsqueeze(0)
                    sample_out["output"][label] = pred
                
                dataset_writer[idx_val] = sample_out
                processed += 1


def predict(
    config_path: str,
    crops: str = "test",
    output_path: str = PREDICTIONS_PATH,
    do_orthoplanes: bool = True,
    overwrite: bool = False,
    search_path: str = SEARCH_PATH,
    raw_name: str = RAW_NAME,
    crop_name: str = CROP_NAME,
):
    config = load_safe_config(config_path)
    classes = config.classes
    batch_size = getattr(config, "batch_size", 8)
    input_array_info = getattr(
        config, "input_array_info", {"shape": (1, 128, 128), "scale": (8, 8, 8)}
    )
    target_array_info = getattr(config, "target_array_info", input_array_info)
    model = config.model

    if getattr(config, "device", None) is not None:
        device = config.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Prediction device: {device}")

    model = model.to(device)
    checkpoint_epoch = get_model(config)
    if checkpoint_epoch is not None:
        print(f"Loaded model checkpoint from epoch: {checkpoint_epoch}")

    disable_orthoplanes = os.getenv("CSC_PRED_DISABLE_ORTHO", "").lower() in ("1", "true", "yes")

    if (
        not disable_orthoplanes
        and do_orthoplanes
        and (array_has_singleton_dim(input_array_info) or is_array_2D(input_array_info, summary=any))
    ):
        print("Using 2.5D orthoplanes inference")
        predict_func = predict_orthoplanes
    else:
        print("Using direct 2D slice-wise inference")
        predict_func = _predict

    input_arrays = {"input": input_array_info}
    target_arrays = {"output": target_array_info}

    if crops == "test":
        test_crops = get_test_crops()
        dataset_writers = []
        for crop in test_crops:
            raw_path = search_path.format(dataset=crop.dataset, name=raw_name)
            target_bounds = {
                "output": {
                    axis: [
                        crop.gt_source.translation[i],
                        crop.gt_source.translation[i]
                        + crop.gt_source.voxel_size[i] * crop.gt_source.shape[i],
                    ]
                    for i, axis in enumerate("zyx")
                },
            }
            dataset_writers.append(
                {
                    "raw_path": raw_path,
                    "target_path": output_path.format(
                        crop=f"crop{crop.id}",
                        dataset=crop.dataset,
                    ),
                    "classes": classes,
                    "input_arrays": input_arrays,
                    "target_arrays": target_arrays,
                    "target_bounds": target_bounds,
                    "overwrite": overwrite,
                    "device": device,
                }
            )
    else:
        crop_list = crops.split(",")
        crop_paths = []
        for i, crop in enumerate(crop_list):
            if (isinstance(crop, str) and crop.isnumeric()) or isinstance(crop, int):
                crop = f"crop{crop}"
                crop_list[i] = crop
            crop_paths.extend(
                glob(
                    search_path.format(
                        dataset="*", name=crop_name.format(crop=crop, label="")
                    ).rstrip(os.path.sep)
                )
            )

        dataset_writers = []
        for crop, crop_path in zip(crop_list, crop_paths):
            raw_path = get_raw_path(crop_path, label="")
            gt_images = {
                array_name: CellMapImage(
                    str(UPath(crop_path) / classes[0]),
                    target_class=classes[0],
                    target_scale=array_info["scale"],
                    target_voxel_shape=array_info["shape"],
                    pad=True,
                    pad_value=0,
                )
                for array_name, array_info in target_arrays.items()
            }
            target_bounds = {
                array_name: image.bounding_box
                for array_name, image in gt_images.items()
            }
            dataset = get_formatted_fields(raw_path, search_path, ["{dataset}"])["dataset"]
            dataset_writers.append(
                {
                    "raw_path": raw_path,
                    "target_path": output_path.format(crop=crop, dataset=dataset),
                    "classes": classes,
                    "input_arrays": input_arrays,
                    "target_arrays": target_arrays,
                    "target_bounds": target_bounds,
                    "overwrite": overwrite,
                    "device": device,
                }
            )

    print(f"Processing {len(dataset_writers)} crops")
    for i, dataset_writer in enumerate(dataset_writers, 1):
        target = dataset_writer["target_path"]
        print(f"\n[{i}/{len(dataset_writers)}] Processing: {target}")
        try:
            predict_func(model, dataset_writer, batch_size)
            print(f"✓ Completed: {target}")
        except Exception as e:
            print(f"✗ Failed: {target}")
            print(f"  Error: {e}")
            continue