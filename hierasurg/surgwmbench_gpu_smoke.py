from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from diffusers import CogVideoXTransformer3DModel
from PIL import Image

from benchmark.surgwmbench.runtime import (
    frame_path_by_index,
    load_annotation,
    read_jsonl,
    resolve_dataset_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tiny HieraSurg/CogVideoX Transformer GPU smoke train on official "
            "SurgWMBench sparse windows. This validates the native CogVideoX model "
            "family without downloading full HieraSurg checkpoints."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-manifest", default="manifests/train.jsonl")
    parser.add_argument("--val-manifest", default="manifests/val.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--prediction-horizon", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--max-clips", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    validate_args(args)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = resolve_dataset_path(args.dataset_root, args.train_manifest)
    val_manifest = resolve_dataset_path(args.dataset_root, args.val_manifest)
    train_windows = load_sparse_windows(
        args.dataset_root,
        train_manifest,
        args.context_frames,
        args.prediction_horizon,
        args.max_clips,
        args.image_size,
    )
    val_windows = load_sparse_windows(
        args.dataset_root,
        val_manifest,
        args.context_frames,
        args.prediction_horizon,
        args.max_clips,
        args.image_size,
    )
    if not train_windows:
        raise RuntimeError(f"No train windows loaded from {train_manifest}")

    model = build_tiny_cogvideox(args.image_size, args.prediction_horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_losses: List[float] = []
    model.train()
    for step in range(args.max_train_steps):
        context_frame, future_frames = train_windows[step % len(train_windows)]
        context_frame = context_frame.to(device)
        future_frames = future_frames.to(device)
        model_input = condition_future_with_context(future_frames, context_frame)
        text_embeddings = torch.zeros(future_frames.shape[0], 4, 16, device=device)
        timestep = torch.zeros(future_frames.shape[0], dtype=torch.long, device=device)
        prediction = model(model_input, text_embeddings, timestep).sample
        loss = torch.nn.functional.mse_loss(prediction, future_frames)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))

    val_loss = None
    if val_windows:
        model.eval()
        with torch.no_grad():
            context_frame, future_frames = val_windows[0]
            context_frame = context_frame.to(device)
            future_frames = future_frames.to(device)
            model_input = condition_future_with_context(future_frames, context_frame)
            text_embeddings = torch.zeros(future_frames.shape[0], 4, 16, device=device)
            timestep = torch.zeros(future_frames.shape[0], dtype=torch.long, device=device)
            prediction = model(model_input, text_embeddings, timestep).sample
            val_loss = float(torch.nn.functional.mse_loss(prediction, future_frames).detach().cpu())

    checkpoint_path = args.output_dir / "hierasurg_surgwmbench_cogvideox_smoke.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "image_size": args.image_size,
                "context_frames": args.context_frames,
                "prediction_horizon": args.prediction_horizon,
                "architecture": "CogVideoXTransformer3DModel",
                "conditioning": "last_context_frame_channel_concat",
            },
        },
        checkpoint_path,
    )
    metrics = {
        "dataset_name": "SurgWMBench",
        "baseline": "hierasurg",
        "model": "HieraSurg",
        "phase": "train",
        "data_track": "sparse_20_anchor",
        "trajectory_target": "sparse_human_anchors",
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "context_frames": args.context_frames,
        "prediction_horizon": args.prediction_horizon,
        "device": str(device),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "num_train_windows": len(train_windows),
        "num_val_windows": len(val_windows),
        "max_train_steps": args.max_train_steps,
        "train_losses": train_losses,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "val_loss": val_loss,
        "checkpoint": str(checkpoint_path),
        "trained_native_architecture": True,
        "full_upstream_cogvideox_two_stage": False,
        "notes": (
            "This smoke trains the CogVideoX Transformer model family used by "
            "HieraSurg on official SurgWMBench sparse windows. It is not a full "
            "HieraSurg semantic-to-map/map-to-video run because that requires "
            "CogVideoX/HieraSurg checkpoints and Cholec-style semantic conditioning."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint_path), "metrics": str(metrics_path)}, indent=2))


def validate_args(args: argparse.Namespace) -> None:
    if args.context_frames != 5:
        raise ValueError("HieraSurg SurgWMBench sparse smoke uses --context-frames 5")
    if args.prediction_horizon not in {5, 10, 15}:
        raise ValueError("--prediction-horizon must be one of 5, 10, 15")
    if args.image_size % 8 != 0:
        raise ValueError("--image-size must be divisible by 8")
    if args.max_train_steps < 1:
        raise ValueError("--max-train-steps must be >= 1")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tiny_cogvideox(image_size: int, prediction_horizon: int) -> CogVideoXTransformer3DModel:
    return CogVideoXTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=6,
        out_channels=3,
        time_embed_dim=32,
        text_embed_dim=16,
        num_layers=1,
        dropout=0.0,
        sample_width=image_size,
        sample_height=image_size,
        sample_frames=prediction_horizon,
        patch_size=2,
        temporal_compression_ratio=1,
        max_text_seq_length=4,
    )


def condition_future_with_context(future_frames: torch.Tensor, context_frame: torch.Tensor) -> torch.Tensor:
    context = context_frame.expand(-1, future_frames.shape[1], -1, -1, -1)
    return torch.cat([future_frames, context], dim=2)


def load_sparse_windows(
    dataset_root: Path,
    manifest: Path,
    context_frames: int,
    prediction_horizon: int,
    max_clips: int,
    image_size: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    windows: List[Tuple[torch.Tensor, torch.Tensor]] = []
    total = context_frames + prediction_horizon
    for row in read_jsonl(manifest, max_clips=max_clips):
        annotation = load_annotation(dataset_root, row)
        anchors = sorted(annotation.get("human_anchors", []), key=lambda item: int(item["anchor_idx"]))
        if len(anchors) != 20 or len(anchors) < total:
            continue
        frame_paths = frame_path_by_index(annotation)
        context = anchors[:context_frames]
        future = anchors[context_frames:total]
        last_context_path = frame_paths[int(context[-1]["local_frame_idx"])]
        future_paths = [frame_paths[int(item["local_frame_idx"])] for item in future]
        context_frame = load_frame_tensor(dataset_root / last_context_path, image_size)
        future_frames = torch.stack(
            [load_frame_tensor(dataset_root / frame_path, image_size) for frame_path in future_paths],
            dim=0,
        )
        windows.append((context_frame.unsqueeze(0).unsqueeze(1), future_frames.unsqueeze(0)))
    return windows


def load_frame_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


if __name__ == "__main__":
    main()
