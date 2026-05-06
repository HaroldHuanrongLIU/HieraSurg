from pathlib import Path
from typing import Any, Dict, Union

import torch
from torch import nn


def trajectory_checkpoint_file(path: Union[str, Path]) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.suffix == ".pt":
        return checkpoint_path
    return checkpoint_path / "trajectory_head.pt"


class SurgWMBenchTrajectoryHead(nn.Module):
    """Predict future normalized trajectory points from context video latents and coordinates."""

    def __init__(
        self,
        latent_channels: int,
        context_anchors: int = 5,
        prediction_anchors: int = 15,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = {
            "latent_channels": latent_channels,
            "context_anchors": context_anchors,
            "prediction_anchors": prediction_anchors,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        }
        self.context_anchors = context_anchors
        self.prediction_anchors = prediction_anchors

        self.latent_norm = nn.LayerNorm(latent_channels)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(context_anchors * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden_dim, prediction_anchors * 2)

    def forward(self, context_latents: torch.Tensor, context_coords_norm: torch.Tensor) -> torch.Tensor:
        if context_latents.ndim != 5:
            raise ValueError(f"Expected context_latents [B,T,C,H,W], got {tuple(context_latents.shape)}")
        if context_coords_norm.shape[1:] != (self.context_anchors, 2):
            raise ValueError(
                f"Expected context_coords_norm [B,{self.context_anchors},2], "
                f"got {tuple(context_coords_norm.shape)}"
            )

        latent_tokens = context_latents.mean(dim=(-1, -2))
        latent_tokens = self.latent_norm(latent_tokens)
        latent_features = self.latent_proj(latent_tokens).mean(dim=1)

        coords = context_coords_norm.to(device=context_latents.device, dtype=context_latents.dtype)
        coord_features = self.coord_proj(coords.reshape(coords.shape[0], -1))
        fused = self.fusion(torch.cat([latent_features, coord_features], dim=-1))
        return self.out(fused).reshape(context_latents.shape[0], self.prediction_anchors, 2).sigmoid()

    def checkpoint_payload(self) -> Dict[str, Any]:
        return {"config": dict(self.config), "state_dict": self.state_dict()}

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        checkpoint_file = trajectory_checkpoint_file(path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), checkpoint_file)

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], map_location: str = "cpu") -> "SurgWMBenchTrajectoryHead":
        checkpoint_file = trajectory_checkpoint_file(path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Missing trajectory head checkpoint: {checkpoint_file}")

        payload = torch.load(checkpoint_file, map_location=map_location, weights_only=False)
        config = payload["config"]
        model = cls(**config)
        model.load_state_dict(payload["state_dict"])
        return model
