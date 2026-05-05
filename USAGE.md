# HieraSurg SurgWMBench Usage

Run all commands from the repository root. The examples use the canonical local
SurgWMBench root:

```bash
/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

To use another copy of the dataset, replace only `--dataset-root`; keep
`--train-manifest`, `--val-manifest`, and `--manifest` relative to that root.

## uv Sync Environment

Create the Python 3.11 environment and install the locked dependencies:

```bash
uv sync
```

Optional labeler dependencies:

```bash
uv sync --extra labeler
```

Verify the synced Python and PyTorch versions:

```bash
uv run python -c "import sys, torch, torchvision; print(sys.version); print(torch.__version__, torchvision.__version__)"
```

Validate the 20-anchor loader before training:

```bash
uv run python src/tools/validate_surgwmbench_anchor_loader.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/train.jsonl \
  --num-samples 8
```

## Single-GPU Training

This trains one model conditioned on anchors 1-5 and supervised on anchors
6-20. Evaluation later reports horizons 6-10, 6-15, and 6-20 from the same
checkpoint.

```bash
uv run accelerate launch --num_processes 1 \
  src/finetune/train_surgwmbench_anchor_i2v.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --pretrained_model_name_or_path /path/to/cogvideox-or-hierasurg-base \
  --output_dir outputs/surgwmbench_anchor_i2v \
  --height 288 \
  --width 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --enable_slicing \
  --enable_tiling
```

For a quick smoke run, append:

```bash
--train_limit 2 --max_train_steps 1
```

## Multi-GPU Training

Use Accelerate DDP by setting `--multi_gpu` and matching `--num_processes` to
the number of GPUs. The global batch size is `train_batch_size * num_processes *
gradient_accumulation_steps`.

```bash
uv run accelerate launch --multi_gpu --num_processes 4 \
  src/finetune/train_surgwmbench_anchor_i2v.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --pretrained_model_name_or_path /path/to/cogvideox-or-hierasurg-base \
  --output_dir outputs/surgwmbench_anchor_i2v_ddp \
  --height 288 \
  --width 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --enable_slicing \
  --enable_tiling
```

Resume from the latest checkpoint in the output directory:

```bash
uv run accelerate launch --multi_gpu --num_processes 4 \
  src/finetune/train_surgwmbench_anchor_i2v.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --train-manifest manifests/train.jsonl \
  --val-manifest manifests/val.jsonl \
  --pretrained_model_name_or_path /path/to/cogvideox-or-hierasurg-base \
  --output_dir outputs/surgwmbench_anchor_i2v_ddp \
  --resume_from_checkpoint latest \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --enable_slicing \
  --enable_tiling
```

## Evaluation

Evaluate against original-resolution target frames. The training resize is an
internal model detail; predictions are resized back before metric computation.

```bash
uv run python src/inference/eval_surgwmbench_anchor_i2v.py \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/val.jsonl \
  --pretrained_model_name_or_path /path/to/cogvideox-or-hierasurg-base \
  --checkpoint outputs/surgwmbench_anchor_i2v/checkpoint-final \
  --output_dir outputs/surgwmbench_anchor_i2v_eval \
  --eval-horizons 5 10 15 \
  --mixed_precision bf16 \
  --save-videos
```
