import torch
import torch.nn as nn
import numpy as np

from adflip.data.all_atom_parse import num_residue_tokens
from adflip.model.zoidberg.utils import FourierEmbedding
from adflip.model.zoidberg.transition_block import TransitionBlock
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch


def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


class PositionalEncodings(torch.nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = torch.nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E



class DihedralFeatures(nn.Module):
    def __init__(self, node_embed_dim):
        """
        Embed dihedral angle features.
        adapt from: https://github.com/facebookresearch/esm
        """
        super(DihedralFeatures, self).__init__()
        # 3 dihedral angles; sin and cos of each angle
        node_in = 6
        # Normalization and embedding
        self.node_embedding = nn.Linear(node_in, node_embed_dim, bias=True)

    def forward(self, X):
        """Featurize coordinates as an attributed graph"""
        V = self._dihedrals(X)
        V = self.node_embedding(V)
        return V

    @staticmethod
    def _dihedrals(X, eps=1e-7, return_angles=False):
        # First 3 coordinates are N, CA, C
        X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1, 2), "constant", 0)
        D = D.view((D.size(0), int(D.size(1) / 3), 3))
        phi, psi, omega = torch.unbind(D, -1)

        if return_angles:
            return phi, psi, omega

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

class ProteinFeatures(torch.nn.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=48,
        augment_eps=0.0,
        diheral_features=False,
        secondary_structure=False,
    ):
        """Extract protein features"""
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.diheral_features = diheral_features
        self.secondary_structure = secondary_structure

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = torch.nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = torch.nn.LayerNorm(edge_features)
        if self.diheral_features:
            self.dihedral_encoder = DihedralFeatures(node_features)
        if self.secondary_structure:
            self.ss_encoder = torch.nn.Linear(9, node_features, bias=False)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, input_features):
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]
        if self.secondary_structure:
            ss = input_features["SS"]
            
        # if self.augment_eps > 0 and self.training:
        #     X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        if self.diheral_features:
            hV = self.dihedral_encoder(input_features["X"])
        else:
            hV = None

        if self.secondary_structure:
            h_ss = self.ss_encoder(ss)
        else:
            h_ss = None


        return hV, E, E_idx, None,None,None





class ProteinFeaturesLigand(torch.nn.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        device=None,
        atom_context_num=16,
        use_side_chains=False,
        mpnn_cutoff=False
    ):
        """Extract protein features"""
        super(ProteinFeaturesLigand, self).__init__()

        self.use_side_chains = use_side_chains

        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.mpnn_cutoff = mpnn_cutoff

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = torch.nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = torch.nn.LayerNorm(edge_features)

        self.node_project_down = torch.nn.Linear(
            5 * num_rbf + 64 + 4, node_features, bias=True
        )
        self.norm_nodes = torch.nn.LayerNorm(node_features)

        self.type_linear = torch.nn.Linear(147, 64)

        self.y_nodes = torch.nn.Linear(147, node_features, bias=False)
        self.y_edges = torch.nn.Linear(num_rbf, node_features, bias=False)

        self.norm_y_edges = torch.nn.LayerNorm(node_features)
        self.norm_y_nodes = torch.nn.LayerNorm(node_features)

        self.atom_context_num = atom_context_num

        # the last 32 atoms in the 37 atom representation
        self.side_chain_atom_types = torch.tensor(
            [
                6,
                6,
                6,
                8,
                8,
                16,
                6,
                6,
                6,
                7,
                7,
                8,
                8,
                16,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                8,
                8,
                6,
                7,
                7,
                8,
                6,
                6,
                6,
                7,
                8,
            ],
            device=device,
        )

        self.periodic_table_features = torch.tensor(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                    63,
                    64,
                    65,
                    66,
                    67,
                    68,
                    69,
                    70,
                    71,
                    72,
                    73,
                    74,
                    75,
                    76,
                    77,
                    78,
                    79,
                    80,
                    81,
                    82,
                    83,
                    84,
                    85,
                    86,
                    87,
                    88,
                    89,
                    90,
                    91,
                    92,
                    93,
                    94,
                    95,
                    96,
                    97,
                    98,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                ],
                [
                    0,
                    1,
                    18,
                    1,
                    2,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                ],
                [
                    0,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                ],
            ],
            dtype=torch.long,
            device=device,
        )

    def _make_angle_features(self, A, B, C, Y):
        v1 = A - B
        v2 = C - B
        e1 = torch.nn.functional.normalize(v1, dim=-1)
        e1_v2_dot = torch.einsum("bli, bli -> bl", e1, v2)[..., None]
        u2 = v2 - e1 * e1_v2_dot
        e2 = torch.nn.functional.normalize(u2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        R_residue = torch.cat(
            (e1[:, :, :, None], e2[:, :, :, None], e3[:, :, :, None]), dim=-1
        )

        local_vectors = torch.einsum(
            "blqp, blyq -> blyp", R_residue, Y - B[:, :, None, :]
        )

        rxy = torch.sqrt(local_vectors[..., 0] ** 2 + local_vectors[..., 1] ** 2 + 1e-8)
        f1 = local_vectors[..., 0] / rxy
        f2 = local_vectors[..., 1] / rxy
        rxyz = torch.norm(local_vectors, dim=-1) + 1e-8
        f3 = rxy / rxyz
        f4 = local_vectors[..., 2] / rxyz

        f = torch.cat([f1[..., None], f2[..., None], f3[..., None], f4[..., None]], -1)
        return f

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, input_features):
        

        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]
        device = X.device

        # if self.augment_eps > 0 and self.training:
        #     print("augmenting")
        #     print(self.augment_eps)
        #     X = X + self.augment_eps * torch.randn_like(X)
        #     Y = Y + self.augment_eps * torch.randn_like(Y)
        # remove it as we add noise in GNN model

        B, L, _, _ = X.shape

        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca  # shift from CA

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        # if self.use_side_chains:
        #     xyz_37 = input_features["xyz_37"]
        #     xyz_37_m = input_features["xyz_37_m"]
        #     E_idx_sub = E_idx[:, :, :16]  # [B, L, 15]
        #     mask_residues = input_features["chain_mask"]
        #     xyz_37_m = xyz_37_m * (1 - mask_residues[:, :, None])
        #     R_m = gather_nodes(xyz_37_m[:, :, 5:], E_idx_sub)

        #     X_sidechain = xyz_37[:, :, 5:, :].view(B, L, -1)
        #     R = gather_nodes(X_sidechain, E_idx_sub).view(
        #         B, L, E_idx_sub.shape[2], -1, 3
        #     )
        #     R_t = self.side_chain_atom_types[None, None, None, :].repeat(
        #         B, L, E_idx_sub.shape[2], 1
        #     )

        #     # Side chain atom context
        #     R = R.view(B, L, -1, 3)  # coordinates
        #     R_m = R_m.view(B, L, -1)  # mask
        #     R_t = R_t.view(B, L, -1)  # atom types

        #     # Ligand atom context
        #     Y = torch.cat((R, Y), 2)  # [B, L, atoms, 3]
        #     Y_m = torch.cat((R_m, Y_m), 2)  # [B, L, atoms]
        #     Y_t = torch.cat((R_t, Y_t), 2)  # [B, L, atoms]

        #     Cb_Y_distances = torch.sum((Cb[:, :, None, :] - Y) ** 2, -1)
        #     mask_Y = mask[:, :, None] * Y_m
        #     Cb_Y_distances_adjusted = Cb_Y_distances * mask_Y + (1.0 - mask_Y) * 10000.0
        #     _, E_idx_Y = torch.topk(
        #         Cb_Y_distances_adjusted, self.atom_context_num, dim=-1, largest=False
        #     )

        #     Y = torch.gather(Y, 2, E_idx_Y[:, :, :, None].repeat(1, 1, 1, 3))
        #     Y_t = torch.gather(Y_t, 2, E_idx_Y)
        #     Y_m = torch.gather(Y_m, 2, E_idx_Y)
        if "Y" in input_features.keys():

            Y = input_features["Y"]
            Y_m = input_features["Y_m"]
            Y_t = input_features["Y_t"]
            Y_t = Y_t.long()
            
            Y_t_g = self.periodic_table_features[1][Y_t.cpu()]  # group; 19 categories including 0
            Y_t_p = self.periodic_table_features[2][Y_t.cpu()]  # period; 8 categories including 0

            Y_t_g_1hot_ = torch.nn.functional.one_hot(Y_t_g, 19).to(device)  # [B, L, M, 19]
            Y_t_p_1hot_ = torch.nn.functional.one_hot(Y_t_p, 8).to(device)  # [B, L, M, 8]
            Y_t_1hot_ = torch.nn.functional.one_hot(Y_t, 120).to(device)  # [B, L, M, 120]

            Y_t_1hot_ = torch.cat(
                [Y_t_1hot_, Y_t_g_1hot_, Y_t_p_1hot_], -1
            )  # [B, L, M, 147]
            Y_t_1hot = self.type_linear(Y_t_1hot_.float())

            D_N_Y = self._rbf(
                torch.sqrt(torch.sum((N[:, :, None, :] - Y) ** 2, -1) + 1e-6)
            )  # [B, L, M, num_bins]
            D_Ca_Y = self._rbf(
                torch.sqrt(torch.sum((Ca[:, :, None, :] - Y) ** 2, -1) + 1e-6)
            )
            D_C_Y = self._rbf(torch.sqrt(torch.sum((C[:, :, None, :] - Y) ** 2, -1) + 1e-6))
            D_O_Y = self._rbf(torch.sqrt(torch.sum((O[:, :, None, :] - Y) ** 2, -1) + 1e-6))
            D_Cb_Y = self._rbf(
                torch.sqrt(torch.sum((Cb[:, :, None, :] - Y) ** 2, -1) + 1e-6)
            )

            f_angles = self._make_angle_features(N, Ca, C, Y)  # [B, L, M, 4]

            D_all = torch.cat(
                (D_N_Y, D_Ca_Y, D_C_Y, D_O_Y, D_Cb_Y, Y_t_1hot, f_angles), dim=-1
            )  # [B,L,M,5*num_bins+5]
            V = self.node_project_down(D_all)  # [B, L, M, node_features]
            V = self.norm_nodes(V)

            Y_edges = self._rbf(
                torch.sqrt(
                    torch.sum((Y[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2, -1) + 1e-6
                )
            )  # [B, L, M, M, num_bins]

            Y_edges = self.y_edges(Y_edges)
            Y_nodes = self.y_nodes(Y_t_1hot_.float())

            Y_edges = self.norm_y_edges(Y_edges)
            Y_nodes = self.norm_y_nodes(Y_nodes)
    
        else:
            Y_nodes = None
            Y_edges = None
            Y_t = None
            V = None
            Y_m = None

        
        return V, E, E_idx, Y_nodes, Y_edges, Y_m










def get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms):
    '''
    from ligandmpnn with batch support
    '''
    device = CB.device
    batch_size = CB.shape[0]
    if (
        mask.shape[0] != batch_size
        or Y.shape[0] != batch_size
        or Y_t.shape[0] != batch_size
        or Y_m.shape[0] != batch_size
    ):
        raise ValueError(
            "Batch size mismatch in get_nearest_neighbours: "
            f"CB={batch_size}, mask={mask.shape[0]}, Y={Y.shape[0]}, "
            f"Y_t={Y_t.shape[0]}, Y_m={Y_m.shape[0]}"
        )
    
    mask_CBY = mask[:, :, None] * Y_m[:, None, :]  # [B,A,C]
    L2_AB = torch.sum((CB[:, :, None, :] - Y[:, None, :, :]) ** 2, -1)  # [B,A,C]
    L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0

    nn_idx = torch.argsort(L2_AB, -1)[:, :, :number_of_ligand_atoms]  # [B,A,K]
    L2_AB_nn = torch.gather(L2_AB, 2, nn_idx)  # [B,A,K]
    D_AB_closest = torch.sqrt(L2_AB_nn[:, :, 0])  # [B,A]

    if number_of_ligand_atoms > nn_idx.shape[2]:
        batch_indices = torch.arange(batch_size, device=device)[:, None, None].expand(-1, CB.shape[1], nn_idx.shape[2])
    else:
        batch_indices = torch.arange(batch_size, device=device)[:, None, None].expand(-1, CB.shape[1], number_of_ligand_atoms)

    Y_tmp = Y[batch_indices, nn_idx]  # [B,A,K,3]
    Y_t_tmp = Y_t[batch_indices, nn_idx]  # [B,A,K]
    Y_m_tmp = Y_m[batch_indices, nn_idx]  # [B,A,K]

    Y = torch.zeros(
        [batch_size, CB.shape[1], number_of_ligand_atoms, 3], dtype=torch.float32, device=device
    )
    Y_t = torch.zeros(
        [batch_size, CB.shape[1], number_of_ligand_atoms], dtype=torch.int32, device=device
    )
    Y_m = torch.zeros(
        [batch_size, CB.shape[1], number_of_ligand_atoms], dtype=torch.int32, device=device
    )

    num_nn_update = Y_tmp.shape[2]
    Y[:, :, :num_nn_update] = Y_tmp
    Y_t[:, :, :num_nn_update] = Y_t_tmp
    Y_m[:, :, :num_nn_update] = Y_m_tmp

    return Y, Y_t, Y_m, D_AB_closest



def compute_ligand_atom(input_dict, batch_dict,number_ligand_atom=10,cutoff_for_score=8.0,mpnn_cutoff=False):
    '''
    ligandmpnn input features
    '''
    device = input_dict["X"].device
    batch_size = input_dict["X"].size(0)
    all_batch_index = batch_dict["batch_index"].long()
    if all_batch_index.numel() == 0:
        return input_dict
    all_batch_index = all_batch_index - all_batch_index.min()

    non_protein_mask = ~batch_dict["is_protein"].bool()
    non_protein_atom_postition = batch_dict["position"][non_protein_mask]
    non_protein_atom_element = batch_dict["element_index"][non_protein_mask]
    non_protein_batch_index = all_batch_index[non_protein_mask]
    if non_protein_batch_index.numel() == 0:
        return input_dict

    Y, Y_mask = to_dense_batch(
        non_protein_atom_postition,
        non_protein_batch_index,
        batch_size=batch_size,
    )
    Y_t, _ = to_dense_batch(
        non_protein_atom_element,
        non_protein_batch_index,
        batch_size=batch_size,
    )
    
    N = input_dict["X"][:, :,0, :]
    CA = input_dict["X"][:,:, 1, :]
    C = input_dict["X"][:, :,2, :]
    b = CA - N
    c = C - CA
    a = torch.cross(b, c, axis=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA


    Y, Y_t, Y_m, D_AB_closest= get_nearest_neighbours(CB, input_dict['mask'], Y, Y_t, Y_mask, number_ligand_atom)
    if mpnn_cutoff:
        mask_XY = (D_AB_closest < cutoff_for_score) * input_dict['mask'] * Y_m[:,:, 0]
        Y_m = mask_XY[:, :, None] * Y_m
        
    input_dict["Y"] = Y
    input_dict["Y_t"] = Y_t
    input_dict["Y_m"] = Y_m


    return input_dict


def dense_input(batch_dict,number_ligand_atom=False,mpnn_cutoff=False):
    batch_size = batch_dict["residue_token"].size(0)
    all_batch_index = batch_dict["batch_index"].long()
    if all_batch_index.numel() == 0:
        raise ValueError("Empty batch_index in dense_input")
    all_batch_index = all_batch_index - all_batch_index.min()

    protein_mask = batch_dict["is_protein"].bool()
    backbone_atom_mask = batch_dict["is_backbone"].bool()
    not_pad_mask = batch_dict["not_pad_mask"].bool()
    backbone_mask = batch_dict.get("backbone_mask", None)
    if backbone_mask is not None:
        backbone_mask = backbone_mask.bool()
    else:
        backbone_mask = torch.ones_like(backbone_atom_mask, dtype=torch.bool)

    center_mask = batch_dict["is_center"].bool() & backbone_mask
    overall_mask = protein_mask & backbone_atom_mask & not_pad_mask & backbone_mask
    backbone_positions = batch_dict["position"][overall_mask]

    flatten_pos = batch_dict["position"][center_mask].unsqueeze(1).repeat(1,4,1).view(-1,3)
    flatten_protein_mask = overall_mask[center_mask].unsqueeze(1).repeat(1,4).view(-1)
    flatten_pos[flatten_protein_mask] = backbone_positions
    flatten_batch_index = all_batch_index[center_mask].unsqueeze(1).repeat(1,4).view(-1)
    flatten_chain_id = batch_dict["chain_id"][center_mask].unsqueeze(1).repeat(1,4).view(-1)
    flatten_residue_index = batch_dict["residue_index"][center_mask].unsqueeze(1).repeat(1,4).view(-1)

    if flatten_batch_index.numel() == 0:
        raise ValueError("No center atoms available after backbone filtering in dense_input")

    padded_backbone_positions, padded_backbone_positions_mask = to_dense_batch(
        flatten_pos, flatten_batch_index, batch_size=batch_size
    )
    padded_backbone_residue_index, _ = to_dense_batch(
        flatten_residue_index, flatten_batch_index, batch_size=batch_size
    )
    padded_backbone_chain_id, _ = to_dense_batch(
        flatten_chain_id, flatten_batch_index, batch_size=batch_size
    )

    input_features = {
        "X": padded_backbone_positions.view(
            padded_backbone_positions.size(0), -1, 4, 3
        ),
        "mask": padded_backbone_positions_mask.view(
            padded_backbone_positions.size(0), -1, 4
        )[:, :, 0].float(),
        "R_idx": padded_backbone_residue_index.view(
            padded_backbone_positions.size(0), -1, 4
        )[:, :, 0],
        "chain_labels": padded_backbone_chain_id.view(
            padded_backbone_positions.size(0), -1, 4
        )[:, :, 0],
    }

    if number_ligand_atom  and batch_dict['residue_token'][~batch_dict['is_protein']].shape[0] > 0:
        input_features= compute_ligand_atom(input_features, batch_dict, number_ligand_atom=number_ligand_atom,mpnn_cutoff=mpnn_cutoff)
    return input_features


class DihedralFeatures(nn.Module):
    def __init__(self, node_embed_dim):
        """
        Embed dihedral angle features.
        adapt from: https://github.com/facebookresearch/esm
        """
        super(DihedralFeatures, self).__init__()
        # 3 dihedral angles; sin and cos of each angle
        node_in = 6
        # Normalization and embedding
        self.node_embedding = nn.Linear(node_in, node_embed_dim, bias=True)

    def forward(self, X):
        """Featurize coordinates as an attributed graph"""
        V = self._dihedrals(X)
        V = self.node_embedding(V)
        return V

    @staticmethod
    def _dihedrals(X, eps=1e-7, return_angles=False):
        # First 3 coordinates are N, CA, C
        X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1, 2), "constant", 0)
        D = D.view((D.size(0), int(D.size(1) / 3), 3))
        phi, psi, omega = torch.unbind(D, -1)

        if return_angles:
            return phi, psi, omega

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features


class BackboneEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_positional_embeddings: int,
        num_rbf: int,
        top_k: int,
        augment_eps: float,
        backbone_diheral: bool = False,
        number_ligand_atom: int = 0,
        mpnn_cutoff = False,
    ):
        super().__init__()
        self.dim = dim
        self.number_ligand_atom =number_ligand_atom
        self.residue_token_embedding = nn.Embedding(num_residue_tokens, hidden_dim) #embedding all residue type include ligand
        self.timestep_emb = FourierEmbedding(hidden_dim)
        self.mpnn_cutoff = mpnn_cutoff
        if number_ligand_atom:
            self.edge_encoder = ProteinFeaturesLigand(
                dim,
                dim,
                num_positional_embeddings=num_positional_embeddings,
                num_rbf=num_rbf,
                top_k=top_k,
                augment_eps=augment_eps,
                atom_context_num = number_ligand_atom,          
            )
        else:
            self.edge_encoder = ProteinFeatures(
                dim,
                dim,
                num_positional_embeddings=num_positional_embeddings,
                num_rbf=num_rbf,
                top_k=top_k,
                augment_eps=augment_eps,
            )
        if backbone_diheral:
            self.dihedral_encoder = DihedralFeatures(hidden_dim)
            self.proj = TransitionBlock(dim, input_dim=2 * hidden_dim)
        else:
            self.proj = TransitionBlock(dim, input_dim=1 * hidden_dim)

    def forward(self, batch_dict: dict):

        batch_size = batch_dict["residue_token"].size(0)
        all_batch_index = batch_dict["batch_index"].long()
        if all_batch_index.numel() == 0:
            raise ValueError("Empty batch_index in BackboneEncoder.forward")
        all_batch_index = all_batch_index - all_batch_index.min()

        center_mask = batch_dict["is_center"].bool()
        if "backbone_mask" in batch_dict:
            center_mask = center_mask & batch_dict["backbone_mask"].bool()
        residue_token = batch_dict["residue_token"][center_mask]
        batch_idx = all_batch_index[center_mask]
        if batch_idx.numel() == 0:
            raise ValueError("No center residues available after backbone filtering")
        padded_residue_token, padded_residue_mask = to_dense_batch(
            residue_token, batch_idx, batch_size=batch_size
        )
        B, N = padded_residue_token.size()
        input_features = dense_input(batch_dict, number_ligand_atom=self.number_ligand_atom,mpnn_cutoff = self.mpnn_cutoff)

        # time_emb = self.timestep_emb(timestep.repeat(1,N))
        residue_context_embedding, E, E_idx, Y_nodes, Y_edges, Y_m = self.edge_encoder(input_features)
        residue_type_embedding = self.residue_token_embedding(padded_residue_token)
        if hasattr(self, "dihedral_encoder"):
            V = self.proj(
                torch.cat([residue_type_embedding, self.dihedral_encoder(input_features["X"])], dim=-1)
            )
        else:
            V = self.proj(residue_type_embedding)

        return V, E, E_idx, input_features["mask"].bool(),residue_context_embedding,Y_nodes, Y_edges, Y_m 
