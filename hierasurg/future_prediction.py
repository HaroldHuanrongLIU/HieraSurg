from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from benchmark.surgwmbench import BaselineSpec, run_cli


def predict_native_hierasurg_frames(
    dataset_root: Path,
    window: object,
    args: argparse.Namespace,
) -> List[np.ndarray]:
    request_path = write_native_request(dataset_root, window, args)
    missing = missing_required_assets()
    if Path(str(args.checkpoint)).suffix == ".json" or missing:
        missing_text = ", ".join(f"{key}={value}" for key, value in missing.items()) or "model paths"
        raise RuntimeError(
            "HieraSurg native prediction requires two CogVideoX/HieraSurg model paths, "
            "a Cholec-style validation data path, and annotations. "
            f"Missing: {missing_text}. Wrote native request to {request_path}."
        )

    raise RuntimeError(
        "HieraSurg native request preparation is implemented, but automated generation "
        "is not launched yet because full_pipeline_inference_i2v.py expects Cholec-style "
        "phase/triplet annotations and segmentation-map conditioning that are not present "
        f"in SurgWMBench manifests. Use {request_path} to build the required conditioning "
        "assets from official data before running the upstream full pipeline."
    )


def write_native_request(dataset_root: Path, window: object, args: argparse.Namespace) -> Path:
    init_frame_path = dataset_root / window.context_frame_paths[-1]
    with Image.open(init_frame_path) as image:
        width, height = image.size
    predicted_coords = predict_trajectory(
        window.context_coords,
        int(args.prediction_horizon),
        str(args.trajectory_predictor),
    )
    request_dir = args.output.parent / "native_requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / f"{safe_id(window.clip_id)}_h{args.prediction_horizon}.json"
    payload: Dict[str, Any] = {
        "dataset_name": "SurgWMBench",
        "baseline": "hierasurg",
        "model": "HieraSurg",
        "clip_id": window.clip_id,
        "data_track": window.data_track,
        "prediction_task": args.prediction_task,
        "context_frames": args.context_frames,
        "prediction_horizon": args.prediction_horizon,
        "init_frame_path": str(init_frame_path),
        "context_indices": window.context_indices,
        "future_indices": window.future_indices,
        "context_coords_norm": window.context_coords,
        "predicted_future_coords_norm": predicted_coords,
        "trajectory_hint_px": [
            [float(point[0]) * width, float(point[1]) * height] for point in predicted_coords
        ],
        "image_width": width,
        "image_height": height,
        "semantic_conditioning": {
            "phase": None,
            "triplet": None,
            "segmentation_maps": None,
            "reason": "SurgWMBench manifests provide sparse instrument anchors, not Cholec phase/triplet labels.",
        },
        "required_env": required_assets(),
        "native_entrypoint": "src/inference/full_pipeline_inference_i2v.py",
        "command_template": [
            "python",
            "src/inference/full_pipeline_inference_i2v.py",
            "--pretrained_model_name_or_path",
            "$HIERASURG_SEGPRED_MODEL",
            "--second_step_model_path",
            "$HIERASURG_MAP2VID_MODEL",
            "--val_data_path",
            "$HIERASURG_VAL_DATA_PATH",
            "--annotations_path",
            "$HIERASURG_ANNOTATIONS_PATH",
            "--max_num_frames",
            str(max(17, int(args.prediction_horizon) + 1)),
            "--output_dir",
            str(args.output.parent / "native_hierasurg_outputs"),
        ],
        "notes": (
            "HieraSurg decouples semantic-map generation from map-to-video generation. "
            "This request records the SurgWMBench init frame and instrument trajectory "
            "hint, but full native execution requires separate semantic conditioning assets."
        ),
    }
    request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return request_path


def predict_trajectory(
    context_coords: Sequence[Sequence[float]],
    horizon: int,
    predictor: str,
) -> List[List[float]]:
    context = np.asarray(context_coords, dtype=np.float64)
    if predictor == "copy_last" or len(context) < 2:
        step = np.zeros(2, dtype=np.float64)
    else:
        step = context[-1] - context[-2]
    start = context[-1]
    predictions = [np.clip(start + step * (idx + 1), 0.0, 1.0).tolist() for idx in range(horizon)]
    return [[float(item[0]), float(item[1])] for item in predictions]


def required_assets() -> Dict[str, str]:
    return {
        "HIERASURG_SEGPRED_MODEL": os.environ.get("HIERASURG_SEGPRED_MODEL", ""),
        "HIERASURG_MAP2VID_MODEL": os.environ.get("HIERASURG_MAP2VID_MODEL", ""),
        "HIERASURG_VAL_DATA_PATH": os.environ.get("HIERASURG_VAL_DATA_PATH", ""),
        "HIERASURG_ANNOTATIONS_PATH": os.environ.get("HIERASURG_ANNOTATIONS_PATH", ""),
    }


def missing_required_assets() -> Dict[str, str]:
    missing: Dict[str, str] = {}
    for key, value in required_assets().items():
        if not value or not Path(value).exists():
            missing[key] = value
    return missing


def safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


SPEC = BaselineSpec(
    baseline="hierasurg",
    model="HieraSurg",
    native_entrypoint="src/inference/full_pipeline_inference_i2v.py",
    native_train_entrypoint="src/finetune/train_cogvideox_cholec_segpred_i2v_full.py",
    native_frame_predictor=(
        "SurgWMBench-to-HieraSurg native request preparation is implemented. Full "
        "generation still needs CogVideoX/HieraSurg checkpoints plus Cholec-style "
        "phase/triplet and segmentation-map conditioning assets."
    ),
    native_frame_predictor_fn=predict_native_hierasurg_frames,
    notes=(
        "HieraSurg decouples semantic-map generation and map-to-video generation. "
        "This adapter preserves SurgWMBench split/track semantics, writes native request "
        "artifacts, and exposes a schema-compatible entrypoint pending checkpoint and "
        "semantic conditioning setup."
    ),
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_cli(SPEC, argv)


if __name__ == "__main__":
    main()
