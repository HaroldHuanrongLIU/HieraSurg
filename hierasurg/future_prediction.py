from __future__ import annotations

from typing import Optional, Sequence

from benchmark.surgwmbench import BaselineSpec, run_cli


SPEC = BaselineSpec(
    baseline="hierasurg",
    model="HieraSurg",
    native_entrypoint="src/inference/full_pipeline_inference_i2v.py",
    native_train_entrypoint="src/finetune/train_cogvideox_cholec_segpred_i2v_full.py",
    native_frame_predictor=(
        "upstream inference expects CogVideoX/HieraSurg checkpoints, segmentation-map "
        "conditioning, and Cholec-style annotations; SurgWMBench manifest mapping is "
        "scaffolded but native execution is not enabled by smoke tests."
    ),
    notes=(
        "HieraSurg decouples semantic-map generation and map-to-video generation. "
        "This adapter preserves SurgWMBench split/track semantics and exposes a "
        "schema-compatible entrypoint pending checkpoint and conditioning setup."
    ),
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_cli(SPEC, argv)


if __name__ == "__main__":
    main()
