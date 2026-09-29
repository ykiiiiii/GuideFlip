import torch
import torch.nn as nn


def knn_graph(x, k, batch=None, loop=True, flow="source_to_target"):
    """Construct a k-nearest-neighbour graph with PyTorch cdist and topk.

    The distance matrix requires O(N²) memory per batch item. Equidistant
    neighbours may be selected differently from torch_cluster.

    Returns [source(neighbour), target(query)] in query-major order for reshaping
    to (B, N, k). Only flow="source_to_target" is supported.
    """
    if flow != "source_to_target":
        raise NotImplementedError(
            f"knn_graph here returns [source, target] and only implements "
            f"flow='source_to_target', not {flow!r}"
        )
    device = x.device
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    src, dst = [], []
    for b in torch.unique(batch):
        idx = (batch == b).nonzero(as_tuple=True)[0]      # global indices, sorted
        pos = x[idx]
        distance = torch.cdist(pos, pos)
        # Without self-loops a point has one fewer neighbour to offer, and asking for
        # k of them when only k-1 exist returns the masked diagonal -- the self-loop
        # that was just excluded, at a distance of 1e9.
        available = pos.shape[0] if loop else pos.shape[0] - 1
        if not loop:
            distance = distance + torch.eye(distance.shape[0], device=device) * 1e9
        kk = min(int(k), available)
        if kk <= 0:
            continue
        neighbours = distance.topk(kk, dim=-1, largest=False).indices
        query = idx.view(-1, 1).expand(-1, kk).reshape(-1)
        src.append(idx[neighbours.reshape(-1)])
        dst.append(query)
    if not src:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.stack([torch.cat(src), torch.cat(dst)], dim=0)


class FourierEmbedding(nn.Module):
    def __init__(self, dim):
        """
        AlphaFold3's Fourier Embedding
        """
        super().__init__()
        self.dim = dim
        self.register_buffer("weight", torch.randn(dim))
        self.register_buffer("bias", torch.randn(dim))

    def forward(self, x):
        return torch.cos(2 * torch.pi * (x[..., None] * self.weight + self.bias))


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int, condition_dim: int = None, eps: float = 1e-6):
        super().__init__()
        if condition_dim is None:
            condition_dim = dim
        self.eps = eps
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False, bias=False, eps=eps)
        self.ln2 = nn.LayerNorm(condition_dim, bias=False, eps=eps)
        self.linear_nb = nn.Linear(condition_dim, dim, bias=False)
        self.linear_sig = nn.Linear(condition_dim, dim)

        nn.init.constant_(self.linear_sig.bias, -2.0)

    def forward(self, x, condition):
        x = self.ln1(x)
        y = self.ln2(condition)
        b = torch.sigmoid(self.linear_sig(y)) * x + self.linear_nb(y)
        return b


class FrameAveraging(nn.Module):

    def __init__(self):
        super().__init__()
        self.ops = torch.tensor(
            [[i, j, k] for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]]
        )

    def create_frame(self, X, mask):
        device = X.device
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask  # [B,N,3]
        C = torch.bmm(X.transpose(1, 2), X)  # [B,3,3] (Cov)
        _, V = torch.linalg.eigh(C.detach(), UPLO="U")  # [B,3,3]
        self.ops = self.ops.to(device)
        F_ops = self.ops.unsqueeze(1).unsqueeze(0) * V.unsqueeze(
            1
        )  # [1,8,1,3] x [B,1,3,3] -> [B,8,3,3]
        h = torch.einsum(
            "boij,bpj->bopi", F_ops.transpose(2, 3), X
        )  # transpose is inverse [B,8,N,3]
        h = h.view(X.size(0) * 8, X.size(1), 3)
        return h, F_ops.detach(), center

    def invert_frame(self, X, mask, F_ops, center):
        X = torch.einsum("boij,bopj->bopi", F_ops, X)
        X = X.mean(dim=1)  # frame averaging
        X = X + center.unsqueeze(1)
        return X * mask.unsqueeze(-1)


def transform_residue_index(residue_index, atom_mask):
    B, N = residue_index.shape
    device = residue_index.device

    # Create a tensor to hold the transformed indices
    transformed_index = torch.full_like(residue_index, -1)

    for b in range(B):
        # Get the residue indices and mask for this batch
        batch_indices = residue_index[b]
        batch_mask = atom_mask[b] & (batch_indices != -1)  # Exclude -1 indices

        # Find the unique residue indices, respecting order
        unique_indices, inverse_indices = torch.unique_consecutive(
            batch_indices[batch_mask], return_inverse=True
        )

        # Assign the new indices
        transformed_index[b, batch_mask] = inverse_indices

    return transformed_index


def transform_atom_index(atom_index, atom_mask, batch_indices):
    B, N = atom_index.shape
    device = atom_index.device

    # Find the maximum residue index for each batch
    max_atom_index = atom_index.max(dim=1).values
    max_K = torch.zeros_like(max_atom_index)
    max_K[1:] = (torch.cumsum(max_atom_index, dim=0) + 1)[:-1]
    max_K = max_K[:, None]

    # Create batch indices
    batch_indices = torch.ones(B, N, device=device, dtype=torch.long)

    # Create a linear index for each residue
    linear_index = batch_indices * max_K + atom_index

    # Create a mask for valid atoms (not masked and have a valid residue index)
    valid_mask = atom_mask & (atom_index >= 0)

    linear_index = torch.where(
        valid_mask, linear_index, torch.full_like(linear_index, -1)
    )
    return linear_index


def gather_residue_average_from_atoms(x, residue_index, atom_mask):
    B, N, D = x.shape
    device = x.device
    # Apply atom mask to x
    x_masked = x * atom_mask.unsqueeze(-1)
    # Find the maximum residue index for each batch
    max_residue_index = residue_index.max()

    max_K = max_residue_index.item() + 1

    # Create batch indices
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)

    # Create a linear index for each residue
    linear_index = batch_indices * max_K + residue_index

    # Create a mask for valid atoms (not masked and have a valid residue index)
    valid_mask = atom_mask & (residue_index >= 0)

    # Sum the vectors for each residue and count the atoms
    residue_sum = torch.zeros(B * max_K, D, device=device)
    residue_count = torch.zeros(B * max_K, 1, device=device)

    residue_sum.index_add_(0, linear_index[valid_mask], x_masked[valid_mask])
    residue_count.index_add_(
        0, linear_index[valid_mask], torch.ones(valid_mask.sum(), 1, device=device)
    )

    # Calculate the average
    residue_avg = residue_sum / (
        residue_count + 1e-10
    )  # Add small epsilon to avoid division by zero

    # Reshape to [B, max_K, D]
    residue_avg = residue_avg.view(B, max_K, D)

    # Create residue mask
    residue_mask = residue_count.view(B, max_K) > 0

    return residue_avg, residue_mask


def scatter_residue_feature_over_atoms(residue_avg, residue_index, atom_mask):
    B, K, D = residue_avg.shape
    N = residue_index.shape[1]
    device = residue_avg.device

    # Create a tensor to hold the reconstructed atom features
    x_reconstructed = torch.zeros((B, N, D), device=device)

    # Create batch indices
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)

    # Use advanced indexing to distribute the residue averages back to atoms
    x_reconstructed = residue_avg[batch_indices, residue_index]

    # Apply the atom mask
    x_reconstructed = x_reconstructed * atom_mask.unsqueeze(-1)

    return x_reconstructed


def find_first_padded(not_pad_mask):
    """Return the first padded index, or the sequence length if none is padded."""
    # Find the first False in each sequence
    first_padded = (~not_pad_mask).long().argmax(dim=1)

    # If a sequence has no padding, argmax will return 0
    # We need to set it to the length of the sequence
    seq_lengths = not_pad_mask.sum(dim=1)
    no_padding = (first_padded == 0) & (seq_lengths == not_pad_mask.shape[1])
    first_padded[no_padding] = not_pad_mask.shape[1]

    return first_padded


class StartEndPad(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.start = nn.Parameter(torch.randn(dim))
        self.end = nn.Parameter(torch.randn(dim))

    def forward(self, x, protein_mask):
        B, N, D = x.shape
        x = nn.functional.pad(x, (0, 0, 1, 1), value=0)  # shape: [B, N+1, D]
        protein_mask = nn.functional.pad(
            protein_mask, (0, 1), value=False
        )  # shape: [B, N+1]
        first_padded = find_first_padded(protein_mask)
        x[:, 0] = self.start[None].repeat(B, 1)
        x[torch.arange(B), first_padded] = self.end[None].repeat(B, 1)
        return x


# timm SwiGLU
class SwiGLULayer(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = [bias, bias]
        drop_probs = [drop, drop]

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
