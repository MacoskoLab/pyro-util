from typing import Sequence, Tuple, Type

import torch
import torch.nn as nn

from pyro.distributions.util import broadcast_shape

T = torch.Tensor


def make_fc(
    dims: Sequence[int], activation: Type[nn.Module] = nn.ReLU, batch_norm: bool = True
) -> nn.Module:
    """Helper function for creating a fully connected neural network. Each layer is made up of
    an optional BatchNorm, an activation function, and a linear layer.

    :param dims: The size of the layers in the network
    :param activation: Activation layer to use
    :param batch_norm: Whether to use nn.BatchNorm1d before activation
    :return: nn.Sequential containing all the layers
    """

    layers = []

    for in_dim, out_dim in zip(dims, dims[1:]):
        if batch_norm:
            layers.append(nn.BatchNorm1d(in_dim))
        layers.append(activation())
        layers.append(nn.Linear(in_dim, out_dim))

    return nn.Sequential(*layers)


def split_in_half(t: T) -> Tuple[T, T]:
    """Splits a tensor in half along the final dimension"""
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)


def broadcast_inputs(input_args):
    """Helper for broadcasting inputs to neural net"""
    shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
    input_args = [s.expand(shape) for s in input_args]
    return input_args
