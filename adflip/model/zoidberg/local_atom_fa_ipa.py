import torch
import torch.nn as nn

from adflip.model.zoidberg.transition_block import TransitionBlock
from adflip.model.zoidberg.local_atom_attention import DistanceEmbedding


class LocalAtomFAIPA(nn.Module):
    """
    A frame averaging IPA module that works on atoms, only operating locally.

    This is a ModelAngelo-type IPA that only predicts queries, the keys are the
    local atoms.
    """

    def __init__(
        self, dim: int, num_heads: int, k: int, max_distance_ang: float = 32.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.k = k
        self.max_distance_ang = max_distance_ang
        # self.distance_embedding = DistanceEmbedding(dim, max_distance_ang)
        # self.distance_emb_proj = TransitionBlock(dim, input_dim=dim * k)
        self.q_pred = nn.Linear(dim, self.num_heads * 3)
        self.v_pred = nn.Linear(dim, self.num_heads * self.head_dim)
        self.fc = TransitionBlock(dim, dim)

    # @torch.compile
    def forward(
        self,
        x,
        knn_graph_indices,
        positions,
        frame_positions,
        distance_matrix,
        not_pad_mask,
    ):
        B, N, _ = positions.shape

        # distance_emb = self.distance_embedding(distance_matrix)

        # Apply padding mask
        not_pad_mask_matrix = not_pad_mask.reshape(B * N).float()
        not_pad_mask_matrix = (
            not_pad_mask_matrix[knn_graph_indices[0]]
            * not_pad_mask_matrix[knn_graph_indices[1]]
        )
        not_pad_mask_matrix = not_pad_mask_matrix.reshape(B, N, self.k, 1)
        # distance_emb = not_pad_mask_matrix * distance_emb
        # x = self.distance_emb_proj(distance_emb.reshape(B, N, -1)) + x
        q_pos = self.q_pred(x).reshape(B * N, 1, self.num_heads, 3)
        k_pos = frame_positions.reshape(B * N, 8, 1, 3)
        q_frame = (q_pos + k_pos)[knn_graph_indices[0]]
        k_pos = k_pos[knn_graph_indices[1]]
        qk_dist = torch.norm(q_frame - k_pos, dim=-1).reshape(
            B, N, 8, self.num_heads, self.k
        )
        # Frame averaging
        qk_dist = qk_dist.mean(dim=2)  # B N h k
        # Affinity calculation like ModelAngelo
        qk_affinities = torch.softmax(-qk_dist, dim=-1)
        # Set masked affinities to zero
        qk_affinities = qk_affinities * not_pad_mask_matrix.reshape(B, N, 1, self.k)
        # Renormalize
        qk_affinities = qk_affinities / (qk_affinities.sum(dim=-1, keepdim=True) + 1e-6)

        # Calculate values
        values = (
            self.v_pred(x)
            .reshape(B * N, -1)[knn_graph_indices[1]]
            .reshape(B, N, self.num_heads, self.head_dim, self.k)
        )
        updated_values = torch.einsum("bnhk,bnhdk -> bnhd", qk_affinities, values)
        updated_values = updated_values.reshape(B, N, self.dim)
        x = self.fc(updated_values) + x
        return x
