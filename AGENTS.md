# Repository Guidelines

## Project Structure & Module Organization

HieraSurg is a Python research repository for surgical video diffusion models. Core code lives in `src/`: `src/finetune/` contains CogVideoX training scripts, dataset loaders, and accelerate configs; `src/inference/` contains inference and metric evaluation entrypoints; `src/segcond/` contains segmentation-conditioned model and pipeline code; `src/tools/` contains dataset preparation and visualization utilities. Automatic labeling code is in `labeler/`, assets are in `assets/`, and Cholec split lists are `train_videos.txt` and `val_videos.txt`. There is currently no dedicated `tests/` directory.

## Build, Test, and Development Commands

Create a Python 3.10 environment, install CUDA-matched PyTorch first, then install project dependencies:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Use `requirements_labeler.txt` only for the SAM2/RADIO labeling environment. Common entrypoints:

```bash
python src/tools/cholec_video_extract_parallel.py videos_in videos 1 --with_fix
PYTHONPATH=src python src/inference/evaluate_metrics.py --main_model_path MODEL --model_type cogvideo2b --data_dir DATA
accelerate launch --config_file src/finetune/cfgs/accelerate_config_machine_single_2b.yaml src/finetune/train_cogvideox_cholec_lora.py --pretrained_model_name_or_path MODEL --train_data_path DATA
python labeler/dataset_track.py --sam_weights_folder modules/sam2/checkpoints --dataset_folder videos --visualize
```

## Coding Style & Naming Conventions

Use Python with 4-space indentation and keep imports explicit. Follow the repository's existing module naming style: lowercase snake_case files such as `frame_dataset_hiera.py`, `evaluate_metrics_8fps.py`, and `train_cogvideox_cholec_lora.py`. Prefer small, script-local argument additions over broad rewrites. Keep model, dataset, and pipeline class names descriptive and CamelCase when they mirror Diffusers conventions.

## Testing Guidelines

No formal test framework or coverage target is configured. Before submitting changes, run the narrowest relevant smoke command: import touched modules with `PYTHONPATH=src`, run data utilities on a tiny folder, or run evaluation/training with one batch where supported. If adding reusable loaders or metrics, add `pytest` tests under `tests/` using names like `test_frame_dataset_hiera.py`.

## Commit & Pull Request Guidelines

Recent commits use short, title-style messages such as `Image test` and `Added automatic labeling and documentation`. Keep future commits concise but more specific, for example `Add SurgWMBench frame loader`. Pull requests should include a summary, exact commands run, required dataset/model paths, and any generated visual examples for inference or labeling changes. Link related issues or experiments when available.

## Security & Configuration Tips

Do not commit datasets, generated videos, checkpoints, SAM/RADIO modules, or API keys. Keep large local assets under ignored external paths such as `modules/`, dataset roots, or experiment output directories.
