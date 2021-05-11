from typing import Literal, Sequence, Tuple

import torch
import torch.nn as nn
from pyro.distributions.util import broadcast_shape

from pyro_util.modules.weight_scaling import GammaReLU, WSLinear

T = torch.Tensor
NORM_MODE = Literal["batch_norm", "weight_scaling", None]


def make_fc(dims: Sequence[int], norm_mode: NORM_MODE = "batch_norm") -> nn.Module:
    """Helper function for creating a fully connected neural network.
    Each layer consists of ReLU activation feeding into a linear layer.
    If norm_mode is "batch_norm" then batch normalization is applied before
    activation, while if norm_mode is "weight_scaling" a weight-scaled linear
    layer is used and a scaled ReLU is used instead of the regular ReLU.

    :param dims: The size of the layers in the network
    :param norm_mode: either "batch_norm", "weight_scaling", or None for no norm
    :return: nn.Sequential containing all the layers
    """

    layers = []

    for in_dim, out_dim in zip(dims, dims[1:]):
        if norm_mode == "weight_scaling":
            layers.append(WSLinear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, out_dim))

        if norm_mode == "batch_norm":
            layers.append(nn.BatchNorm1d(out_dim))

        if norm_mode == "weight_scaling":
            layers.append(GammaReLU())
        else:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers[:-1])


def split_in_half(t: T) -> Tuple[T, T]:
    """Splits a tensor in half along the final dimension"""
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)


def broadcast_inputs(input_args):
    """Helper for broadcasting inputs to neural net"""
    shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
    input_args = [s.expand(shape) for s in input_args]
    return input_args
