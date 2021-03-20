from typing import Sequence, Tuple, Type

import torch
import torch.nn as nn

T = torch.Tensor


def make_fc(
    dims: Sequence[int], activation: Type[nn.Module] = nn.ReLU, batch_norm: bool = True
) -> nn.Module:

    layers = []

    for in_dim, out_dim in zip(dims, dims[1:]):
        if batch_norm:
            layers.append(nn.BatchNorm1d(in_dim))
        layers.append(activation())
        layers.append(nn.Linear(in_dim, out_dim))

    return nn.Sequential(*layers)


# Splits a tensor in half along the final dimension
def split_in_half(t: T) -> Tuple[T, T]:
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)
