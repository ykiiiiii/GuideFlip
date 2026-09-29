import torch
import torch.nn as nn

from adflip.data.all_atom_parse import num_residue_tokens, num_element_tokens
from adflip.model.zoidberg.utils import FourierEmbedding
from adflip.model.zoidberg.transition_block import TransitionBlock


class AtomEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.dim = dim
        self.residue_emb = FourierEmbedding(hidden_dim)
        self.atom_emb = FourierEmbedding(hidden_dim)
        self.chain_emb = FourierEmbedding(hidden_dim)
        self.timestep_emb = FourierEmbedding(hidden_dim)
        self.residue_token_emb = nn.Embedding(num_residue_tokens, hidden_dim)
        self.element_token_emb = nn.Embedding(num_element_tokens, hidden_dim)
        self.is_ion_emb = nn.Embedding(2, hidden_dim)
        self.is_protein_emb = nn.Embedding(2, hidden_dim)
        self.is_nucleotide_emb = nn.Embedding(2, hidden_dim)
        self.is_backbone_emb = nn.Embedding(2, hidden_dim)

        self.proj = TransitionBlock(dim, input_dim=9 * hidden_dim)

    def forward(self, batch_dict: dict):
        """Batch dict looks like
        residue_token: torch.Tensor
        residue_index: torch.Tensor
        residue_atom_index: torch.Tensor
        timestep_index: torch.Tensor
        occupancy: torch.Tensor
        bfactor: torch.Tensor
        batch_index: torch.Tensor
        chain_id: torch.Tensor
        position: torch.Tensor
        element_index: torch.Tensor
        atom_name: torch.Tensor
        is_ion: torch.Tensor
        is_protein: torch.Tensor
        is_nucleotide: torch.Tensor
        is_center: torch.Tensor
        is_backbone: torch.Tensor
        not_pad_mask: torch.Tensor
        """
        B, N = batch_dict["residue_token"].size()
        residue_emb = self.residue_emb(batch_dict["residue_index"])
        atom_emb = self.atom_emb(batch_dict["residue_atom_index"])
        chain_emb = self.chain_emb(batch_dict["chain_id"])
        # timestep_emb = self.timestep_emb(timestep.repeat(1,N))
        residue_token_emb = self.residue_token_emb(batch_dict["residue_token"])
        element_token_emb = self.element_token_emb(batch_dict["element_index"])
        is_ion_emb = self.is_ion_emb(batch_dict["is_ion"].long())
        is_protein_emb = self.is_protein_emb(batch_dict["is_protein"].long())
        is_nucleotide_emb = self.is_nucleotide_emb(batch_dict["is_nucleotide"].long())
        is_backbone_emb = self.is_backbone_emb(batch_dict["is_backbone"].long())

        x = torch.cat(
            [
                residue_emb,
                atom_emb,
                chain_emb,
                # timestep_emb,
                residue_token_emb,
                element_token_emb,
                is_ion_emb,
                is_protein_emb,
                is_nucleotide_emb,
                is_backbone_emb,
            ],
            dim=-1,
        )

        return self.proj(x)
