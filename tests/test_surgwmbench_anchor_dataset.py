from pathlib import Path

import pytest
import torch

from finetune.dataset.surgwmbench_anchor_dataset import (
    SurgWMBenchAnchorDataset,
    latent_frame_count,
    pad_anchor_video,
)


DATASET_ROOT = Path("/mnt/hdd1/neurips2026_dataset_track/SurgWMBench")


def test_latent_frame_count_matches_cogvideox_rule():
    assert latent_frame_count(5, 4) == 2
    assert latent_frame_count(20, 4) == 5
    assert latent_frame_count(33, 4) == 9


def test_pad_anchor_video_repeats_last_anchor():
    frames = torch.arange(1 * 3 * 20 * 2 * 2).view(1, 3, 20, 2, 2)
    padded = pad_anchor_video(frames, 33)
    assert padded.shape == (1, 3, 33, 2, 2)
    assert torch.equal(padded[:, :, :20], frames)
    assert torch.equal(padded[:, :, -1], frames[:, :, -1])


@pytest.mark.skipif(not DATASET_ROOT.exists(), reason="SurgWMBench dataset root is not available")
def test_real_surgwmbench_anchor_sample_uses_twenty_human_anchors():
    dataset = SurgWMBenchAnchorDataset(
        dataset_root=str(DATASET_ROOT),
        manifest="manifests/train.jsonl",
        height=64,
        width=64,
        limit=1,
    )
    sample = dataset[0]
    assert sample["anchor_frames"].shape == (3, 20, 64, 64)
    assert sample["context_frames"].shape == (3, 5, 64, 64)
    assert sample["target_frames"].shape == (3, 15, 64, 64)
    assert len(sample["sampled_indices"]) == 20
    assert sample["original_size"] == (1080, 1920)
    assert sample["anchor_frame_paths"][0].endswith(".png")
