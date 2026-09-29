import torch
import torch.nn as nn

from adflip.model.zoidberg.utils import scatter_residue_feature_over_atoms


class NodeToAtomEmbedder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
        )

    def forward(self, x, residue_features, unique_residue_index, not_pad_mask):
        residue_features_scattered = scatter_residue_feature_over_atoms(
            residue_features, unique_residue_index, not_pad_mask
        )
        residue_features_scattered = self.proj(residue_features_scattered)
        return x + residue_features_scattered
