import torch
import torch.nn as nn
import math
import logging

from adflip.model.zoidberg.atom_encoder import AtomEncoder
from adflip.model.zoidberg.local_atom_fa_ipa import LocalAtomFAIPA
from adflip.model.zoidberg.atom_to_node_embedder import AtomToNodeEmbedder
from adflip.model.zoidberg.node_to_atom_embedder import NodeToAtomEmbedder
from adflip.model.zoidberg.backbone_encoder import BackboneEncoder
from adflip.model.zoidberg.context_encoder import ContextBlock
from adflip.model.zoidberg.local_residue_GNN import EncLayer
from adflip.model.zoidberg.transition_block import TransitionBlock
from adflip.model.zoidberg.utils import (
    knn_graph,
    gather_residue_average_from_atoms,
    transform_residue_index,
    FrameAveraging,
)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Zoidberg_GNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        encoder_hidden_dim: int,
        num_blocks: int,
        num_heads: int,
        k: int,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        augment_eps: float = 0.0,
        max_distance_ang: float = 32.0,
        number_ligand_atom: int = 0,
        backbone_diheral: bool = False,
        output_to_esm: bool = False,
        denoiser: bool = True,
        dropout: float = 0.1,
        update_atom: bool = False,
        num_decoder_blocks: int = 0,#using transformer as decoder
        num_tfmr_heads: int = 0,
        num_tfmr_layers: int = 0,
        mpnn_cutoff: bool = False,
        output_dim: int = 20,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.k = k
        self.max_distance_ang = max_distance_ang
        self.number_ligand_atom = number_ligand_atom 
        self.output_to_esm = output_to_esm
        self.denoiser = denoiser
        self.augment_eps = augment_eps
        self.update_atom = update_atom
        self.num_decoder_blocks = num_decoder_blocks
        self.num_tfmr_heads = num_tfmr_heads
        self.num_tfmr_layers = num_tfmr_layers
        self.mpnn_cutoff = mpnn_cutoff
        self.time_embedder = TimestepEmbedder(hidden_dim)
        self.layers = nn.ModuleDict()
        self.layers["atom_encoder"] = AtomEncoder(hidden_dim, encoder_hidden_dim)
        self.backbone_feature = BackboneEncoder(
            hidden_dim,
            encoder_hidden_dim,
            num_positional_embeddings,
            num_rbf,
            top_k=k,
            augment_eps=augment_eps,
            backbone_diheral=backbone_diheral,
            number_ligand_atom = number_ligand_atom,
            mpnn_cutoff = self.mpnn_cutoff
        )
        self.frame_averaging = FrameAveraging()


        if self.number_ligand_atom > 0:
            self.W_v = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.V_C = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.V_C_norm = torch.nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.context_block = nn.ModuleList([ContextBlock(hidden_dim, dropout=dropout) for _ in range(2)])

        for i in range(num_blocks):
            self.layers[f"local_atom_attention_{i}"] = LocalAtomFAIPA(
                hidden_dim, num_heads, k, max_distance_ang=max_distance_ang
            )
            self.layers[f"atom_to_node_embedder_{i}"] = AtomToNodeEmbedder(hidden_dim)
            self.layers[f"gnn_block_{i}"] = EncLayer(
                hidden_dim, 2 * hidden_dim, dropout=dropout, time_embedder=True
            )
            if i < num_blocks - 1 and update_atom:
                self.layers[f"node_to_atom_embedder_{i}"] = NodeToAtomEmbedder(
                    hidden_dim
                )
                self.layers[f"transition_block_{i}"] = TransitionBlock(
                    hidden_dim, input_dim=hidden_dim
                )

        if self.num_decoder_blocks > 0:
            self.decoder_block = nn.ModuleDict()
            for i in range(self.num_decoder_blocks):
                self.decoder_block[f"decoder_block_{i}"] = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=self.num_tfmr_heads,
                    dim_feedforward=num_tfmr_heads * 64,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                ),
                num_layers=self.num_tfmr_layers,
            )



        self.layers["output"] = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch_dict: dict, timestep: torch.Tensor):

        if self.augment_eps > 0.0 and self.training:
            batch_dict["position"] = batch_dict["position"] + torch.randn_like(batch_dict["position"]) * self.augment_eps

        B, N, _ = batch_dict["position"].shape
        positions = batch_dict["position"].reshape(B * N, -1)

        knn_graph_indices = knn_graph(
            positions, k=self.k, batch=batch_dict["batch_index"].flatten(), loop=True
        )
        distance_matrix = torch.norm(
            positions[knn_graph_indices[0]] - positions[knn_graph_indices[1]],
            dim=-1,
        )
        distance_matrix = distance_matrix.reshape(B, N, self.k)
        time_emb = self.time_embedder(timestep * 1000)
        atom_x = self.layers["atom_encoder"](batch_dict)
        center_mask = batch_dict["is_center"].bool()
        if "backbone_mask" in batch_dict:
            center_mask = center_mask & batch_dict["backbone_mask"].bool()
        unique_residue_index = transform_residue_index(
            batch_dict["residue_index"], batch_dict["not_pad_mask"] & center_mask
        )
        residue_x, E, E_idx, residue_x_pad_mask,residue_context_embedding,Y_nodes, Y_edges, Y_m  = self.backbone_feature(batch_dict)
        frame_pos = self.frame_averaging.create_frame(
            batch_dict["position"], batch_dict["not_pad_mask"]
        )[0]
        # Calculate
        for i in range(self.num_blocks):
            atom_x = self.layers[f"local_atom_attention_{i}"](
                atom_x,
                knn_graph_indices,
                batch_dict["position"],
                frame_pos,
                distance_matrix,
                batch_dict["not_pad_mask"],
            )
            residue_x_from_atom, residue_mask = self.layers[
                f"atom_to_node_embedder_{i}"
            ](
                atom_x,
                center_mask,
                unique_residue_index,
                batch_dict["not_pad_mask"],
            )

            if residue_x.shape != residue_x_from_atom.shape:
                raise ValueError(
                    f"Residue shape {residue_x.shape} != {residue_x_from_atom.shape}"
                )
            residue_x, E = self.layers[f"gnn_block_{i}"](
                residue_x,#h_V
                residue_x_from_atom,#h_V_atom
                E,
                E_idx,
                residue_x_pad_mask,
                time=time_emb,
            )  # Double check masking

            if i < self.num_blocks - 1 and self.update_atom:
                atom_x = self.layers[f"node_to_atom_embedder_{i}"](
                    atom_x,
                    residue_x,
                    unique_residue_index,
                    batch_dict["not_pad_mask"],
                )
                atom_x = self.layers[f"transition_block_{i}"](atom_x)

        # Context layer
        
        if self.number_ligand_atom > 0 and batch_dict['residue_token'][~batch_dict['is_protein']].shape[0] > 0: #check has non-protein element
            h_E_context = self.W_v(residue_context_embedding)
            residue_x_context = residue_x   
            for i in range(len(self.context_block)):
                Y_nodes,residue_x_context = self.context_block[i](
                    Y_nodes, Y_edges, Y_m,  residue_x_context, h_E_context,residue_x_pad_mask
                )
            residue_x_context = self.V_C(residue_x_context)
            residue_x = residue_x + self.V_C_norm(self.dropout(residue_x_context))
        else:
            logging.getLogger(__name__).debug("No non-protein residue, skip context block")

        # for i in range(self.num_blocks//2,self.num_blocks):
        #     atom_x = self.layers[f"local_atom_attention_{i}"](
        #         atom_x,
        #         knn_graph_indices,
        #         batch_dict["position"],
        #         frame_pos,
        #         distance_matrix,
        #         batch_dict["not_pad_mask"],
        #     )
        #     residue_x_from_atom, residue_mask = self.layers[
        #         f"atom_to_node_embedder_{i}"
        #     ](
        #         atom_x,
        #         batch_dict["is_center"],
        #         unique_residue_index,
        #         batch_dict["not_pad_mask"],
        #     )

        #     if residue_x.shape != residue_x_from_atom.shape:
        #         raise ValueError(
        #             f"Residue shape {residue_x.shape} != {residue_x_from_atom.shape}"
        #         )
        #     residue_x, E = self.layers[f"gnn_block_{i}"](
        #         residue_x,#h_V
        #         residue_x_from_atom,#h_V_atom
        #         E,
        #         E_idx,
        #         residue_x_pad_mask,
        #         time=time_emb,
        #     )
        #     if i < self.num_blocks - 1 and self.update_atom:
        #         atom_x = self.layers[f"node_to_atom_embedder_{i}"](
        #             atom_x,
        #             residue_x,
        #             unique_residue_index,
        #             batch_dict["not_pad_mask"],
        #         )
        #         atom_x = self.layers[f"transition_block_{i}"](atom_x)

        
        
        
        
        
        #decoder layer
        if self.num_decoder_blocks > 0:#decoder layer
            for i in range(self.num_decoder_blocks):
                residue_x = self.decoder_block[f"decoder_block_{i}"](
                    residue_x, src_key_padding_mask=~residue_x_pad_mask
                )

        # Lastly, just grab information for the protein residues
        # Need to pad start and end tokens (learnable)
        protein_mask, _ = gather_residue_average_from_atoms(
            batch_dict["is_protein"].float().unsqueeze(-1),
            unique_residue_index,
            batch_dict["not_pad_mask"],
        )
        protein_mask = protein_mask.squeeze(-1) > 0.0
        protein_mask = residue_mask * protein_mask
        embedding = residue_x
        logits = self.layers["output"](residue_x)
        flatten_logits = logits[protein_mask]
        return flatten_logits, embedding


if __name__ == "__main__":
    import torch
    from adflip.data.all_atom_parse import get_example_batch

    batch_dict = get_example_batch().__dict__
    batch_dict = {
        k: torch.from_numpy(v) for k, v in batch_dict.items() if k != "atom_name"
    }
    batch_dict = {
        k: v.float() if v.dtype == torch.float64 else v for k, v in batch_dict.items()
    }

    model = Zoidberg_GNN(
        hidden_dim=32,
        encoder_hidden_dim=64,
        num_blocks=3,
        num_heads=4,
        k=4,
        denoiser=True,
    )
    model.eval()
    time_step = torch.randn(batch_dict["batch_index"].size(0), 1)
    logits, embedding = model(batch_dict, time_step)
    print(logits.shape)
