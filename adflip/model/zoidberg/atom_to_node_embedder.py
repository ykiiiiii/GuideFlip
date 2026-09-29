import torch
import torch.nn as nn

from adflip.model.zoidberg.utils import gather_residue_average_from_atoms


class AtomToNodeEmbedder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
        )

    def forward(self, x, is_center, unique_residue_index, not_pad_mask):
        x = self.proj(x)
        node_emb, residue_mask = gather_residue_average_from_atoms(
            x, unique_residue_index, not_pad_mask
        )
        return node_emb, residue_mask
