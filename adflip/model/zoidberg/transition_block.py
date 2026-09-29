import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionBlock(nn.Module):
    def __init__(self, dim: int, input_dim: int = None):
        """
        Similar to AlphaFold3's Conditioned Transition Block
        """
        super().__init__()
        if input_dim is None:
            input_dim = dim
        self.ln = nn.LayerNorm(input_dim)
        self.swish_linear = nn.Linear(input_dim, dim, bias=False)
        self.swish_scale_linear = nn.Linear(input_dim, dim, bias=False)
        self.sigmoid_linear = nn.Linear(input_dim, dim, bias=True)
        self.sigmoid_scale_linear = nn.Linear(dim, dim, bias=False)
        nn.init.constant_(self.sigmoid_linear.bias, -2.0)

    # @torch.compile
    def forward(self, x):
        x = self.ln(x)
        b = F.silu(self.swish_linear(x)) * self.swish_scale_linear(x)
        x = F.sigmoid(self.sigmoid_linear(x)) * self.sigmoid_scale_linear(b)
        return x


# class ConditionedTransitionBlock(nn.Module):
#     def __init__(self, dim: int, input_dim: int = None, condition_dim: int = None):
#         super().__init__()
#         if input_dim is None:
#             input_dim = dim
#         if condition_dim is None:
#             condition_dim = dim
