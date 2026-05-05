import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from finetune.dataset.surgwmbench_anchor_dataset import (
    SurgWMBenchAnchorDataset,
    latent_frame_count,
    pad_anchor_video,
    surgwmbench_anchor_collate,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HieraSurg on SurgWMBench 20-anchor prediction.")
    parser.add_argument("--pretrained_model_name_or_path", required=True, help="Base CogVideoX/HieraSurg model path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir containing transformer/ or a transformer dir.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", default="manifests/val.jsonl")
    parser.add_argument("--output_dir", default="outputs/surgwmbench_anchor_i2v_eval")
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--context-anchors", type=int, default=5)
    parser.add_argument("--prediction-anchors", type=int, default=15)
    parser.add_argument("--model-num-frames", type=int, default=33)
    parser.add_argument("--eval-horizons", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--max-videos", type=int, default=8)
    parser.add_argument("--enable_slicing", action="store_true")
    parser.add_argument("--enable_tiling", action="store_true")
    return parser.parse_args()


def transformer_checkpoint_path(checkpoint: str) -> Path:
    path = Path(checkpoint)
    if (path / "transformer").is_dir():
        return path / "transformer"
    return path


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int,
    patch_size: int,
    attention_head_dim: int,
    device: torch.device,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )
    return freqs_cos.to(device=device), freqs_sin.to(device=device)


def encode_video_latents(vae: AutoencoderKLCogVideoX, video: torch.Tensor) -> torch.Tensor:
    video = video.to(device=vae.device, dtype=vae.dtype)
    encoded = vae.encode(video)
    latent_dist = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded[0]
    return latent_dist.sample() * vae.config.scaling_factor


def decode_video_latents(vae: AutoencoderKLCogVideoX, latents: torch.Tensor) -> torch.Tensor:
    latents = latents.permute(0, 2, 1, 3, 4)
    latents = latents / vae.config.scaling_factor
    return vae.decode(latents.to(dtype=vae.dtype)).sample


def zero_prompt_embeds(batch_size: int, model_config, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(
        (batch_size, model_config.max_text_seq_length, getattr(model_config, "text_embed_dim", 4096)),
        device=device,
        dtype=dtype,
    )


def prepare_extra_step_kwargs(scheduler, generator):
    import inspect

    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    return {"generator": generator} if accepts_generator else {}


@torch.no_grad()
def generate_future_anchors(
    transformer: CogVideoXTransformer3DModel,
    vae: AutoencoderKLCogVideoX,
    scheduler: CogVideoXDPMScheduler,
    context_frames: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size = context_frames.shape[0]
    context_video = context_frames
    repeat_count = args.model_num_frames - args.context_anchors
    pad = context_frames[:, :, -1:].repeat(1, 1, repeat_count, 1, 1)
    conditioning_video = torch.cat([context_video, pad], dim=2).to(device=device, dtype=torch.float32)

    context_latents = encode_video_latents(vae, conditioning_video).to(dtype=dtype).permute(0, 2, 1, 3, 4)
    context_latent_frames = latent_frame_count(args.context_anchors, vae.config.temporal_compression_ratio)
    latents = randn_tensor(context_latents.shape, generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma
    latents[:, :context_latent_frames] = context_latents[:, :context_latent_frames]

    prompt_embeds = zero_prompt_embeds(batch_size, transformer.config, device, dtype)
    timesteps, _ = retrieve_timesteps(scheduler, args.num_inference_steps, device, None)
    extra_step_kwargs = prepare_extra_step_kwargs(scheduler, generator)
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    image_rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=args.height,
            width=args.width,
            num_frames=latents.shape[1],
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            patch_size=transformer.config.patch_size,
            attention_head_dim=transformer.config.attention_head_dim,
            device=device,
        )
        if transformer.config.use_rotary_positional_embeddings
        else None
    )

    old_pred_original_sample = None
    for i, t in enumerate(tqdm(timesteps, desc="Denoising", leave=False)):
        latent_model_input = scheduler.scale_model_input(latents, t)
        latent_model_input[:, :context_latent_frames] = context_latents[:, :context_latent_frames]
        timestep = t.expand(batch_size)
        noise_pred = transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0].float()

        if not isinstance(scheduler, CogVideoXDPMScheduler):
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        else:
            latents, old_pred_original_sample = scheduler.step(
                noise_pred,
                old_pred_original_sample,
                t,
                timesteps[i - 1] if i > 0 else None,
                latents,
                **extra_step_kwargs,
                return_dict=False,
            )
        latents = latents.to(dtype)
        latents[:, :context_latent_frames] = context_latents[:, :context_latent_frames]

    decoded = decode_video_latents(vae, latents).clamp(-1, 1)
    return decoded[:, :, args.context_anchors : args.context_anchors + args.prediction_anchors]


def tensor_to_uint8(frame: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
    frame = ((frame.float().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
    image = Image.fromarray((frame * 255).astype(np.uint8))
    image = image.resize((size[1], size[0]), Image.Resampling.BICUBIC)
    return np.asarray(image).astype(np.uint8)


def load_original_uint8(dataset_root: str, relative_path: str) -> np.ndarray:
    with Image.open(Path(dataset_root) / relative_path) as image:
        return np.asarray(image.convert("RGB")).astype(np.uint8)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


def simple_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = ((pred - mu_x) ** 2).mean()
    sigma_y = ((target - mu_y) ** 2).mean()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(value.item())


def make_lpips(device: torch.device):
    try:
        import lpips

        return lpips.LPIPS(net="alex").to(device).eval()
    except Exception:
        return None


def compute_frame_metrics(
    pred_uint8: np.ndarray,
    target_uint8: np.ndarray,
    lpips_model,
    device: torch.device,
) -> Dict[str, Optional[float]]:
    pred = torch.from_numpy(pred_uint8).float().permute(2, 0, 1) / 255.0
    target = torch.from_numpy(target_uint8).float().permute(2, 0, 1) / 255.0
    result: Dict[str, Optional[float]] = {
        "psnr": psnr(pred, target),
        "ssim": simple_ssim(pred, target),
        "lpips": None,
    }
    if lpips_model is not None:
        with torch.no_grad():
            pred_lp = pred.unsqueeze(0).to(device) * 2 - 1
            target_lp = target.unsqueeze(0).to(device) * 2 - 1
            result["lpips"] = float(lpips_model(pred_lp, target_lp).item())
    return result


def save_video(path: Path, frames: List[np.ndarray], fps: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=10) as writer:
        for frame in frames:
            writer.append_data(frame)


def mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def main() -> None:
    args = get_args()
    if args.context_anchors + args.prediction_anchors != 20:
        raise ValueError("Expected 5 context anchors plus 15 prediction anchors.")
    if args.model_num_frames < 20 or (args.model_num_frames - 1) % 4 != 0:
        raise ValueError("--model-num-frames must be >=20 and satisfy (N - 1) % 4 == 0.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.mixed_precision]

    dataset = SurgWMBenchAnchorDataset(
        dataset_root=args.dataset_root,
        manifest=args.manifest,
        height=args.height,
        width=args.width,
        context_anchors=args.context_anchors,
        prediction_anchors=args.prediction_anchors,
        limit=args.num_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=surgwmbench_anchor_collate,
    )

    transformer = CogVideoXTransformer3DModel.from_pretrained(transformer_checkpoint_path(args.checkpoint), torch_dtype=dtype)
    vae = AutoencoderKLCogVideoX.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32)
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()
    transformer.to(device=device, dtype=dtype).eval()
    vae.to(device=device, dtype=torch.float32).eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    lpips_model = make_lpips(device)
    metrics_by_horizon: Dict[str, Dict[str, List[Optional[float]]]] = {
        f"horizon_{h}": {"psnr": [], "ssim": [], "lpips": []} for h in args.eval_horizons
    }
    samples: List[Dict[str, Any]] = []
    saved_videos = 0

    for batch in tqdm(loader, desc="Evaluating"):
        context_frames = batch["context_frames"].to(device=device, dtype=torch.float32)
        pred_targets = generate_future_anchors(transformer, vae, scheduler, context_frames, args, device, dtype, generator)

        for b in range(pred_targets.shape[0]):
            original_size = batch["original_size"][b]
            pred_original = [
                tensor_to_uint8(pred_targets[b, :, frame_idx], original_size)
                for frame_idx in range(args.prediction_anchors)
            ]
            target_original = [
                load_original_uint8(args.dataset_root, rel_path) for rel_path in batch["target_frame_paths"][b]
            ]

            sample_metrics: Dict[str, Dict[str, Optional[float]]] = {}
            for horizon in args.eval_horizons:
                horizon_key = f"horizon_{horizon}"
                frame_metrics = [
                    compute_frame_metrics(pred_original[i], target_original[i], lpips_model, device)
                    for i in range(horizon)
                ]
                aggregate = {
                    "psnr": mean_or_none([m["psnr"] for m in frame_metrics]),
                    "ssim": mean_or_none([m["ssim"] for m in frame_metrics]),
                    "lpips": mean_or_none([m["lpips"] for m in frame_metrics]),
                }
                sample_metrics[horizon_key] = aggregate
                for metric_name, value in aggregate.items():
                    metrics_by_horizon[horizon_key][metric_name].append(value)

            if args.save_videos and saved_videos < args.max_videos:
                pred_path = output_dir / "videos" / f"pred_{saved_videos:04d}.mp4"
                target_path = output_dir / "videos" / f"target_{saved_videos:04d}.mp4"
                save_video(pred_path, pred_original)
                save_video(target_path, target_original)
                saved_videos += 1

            samples.append(
                {
                    "patient_id": batch["patient_id"][b],
                    "trajectory_id": batch["trajectory_id"][b],
                    "difficulty": batch["difficulty"][b],
                    "sampled_indices": batch["sampled_indices"][b],
                    "target_frame_paths": batch["target_frame_paths"][b],
                    "metrics": sample_metrics,
                }
            )

    aggregate_metrics = {
        horizon: {metric: mean_or_none(values) for metric, values in metric_values.items()}
        for horizon, metric_values in metrics_by_horizon.items()
    }
    result = {
        "dataset_name": "SurgWMBench",
        "model": "HieraSurg anchor I2V",
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "context_anchors": args.context_anchors,
        "prediction_anchors": args.prediction_anchors,
        "eval_horizons": args.eval_horizons,
        "metric_resolution": "original",
        "generated_resolution": [args.height, args.width],
        "metrics": aggregate_metrics,
        "lpips_available": lpips_model is not None,
        "num_samples": len(samples),
        "samples": samples,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"metrics": aggregate_metrics, "output": str(output_dir / "metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
