from typing import Tuple

import torch
import torch.nn as nn
from pyro.distributions.util import broadcast_shape

from pyro_util.modules.weight_scaling import GammaReLU, WSLinear

T = torch.Tensor


def make_ws_fc(*dims: int) -> nn.Module:
    """Helper function for creating a fully connected neural network.
    This version uses weight-scaled linear layers and gamma-scaled ReLU

    :param dims: The size of the layers in the network (at least 2)
    :return: nn.Sequential containing all the layers
    """

    layers = [WSLinear(dims[0], dims[1])]

    for in_dim, out_dim in zip(dims[1:], dims[2:]):
        layers.append(GammaReLU())
        layers.append(WSLinear(in_dim, out_dim))

    return nn.Sequential(*layers)


def make_bn_fc(*dims: int) -> nn.Module:
    """Helper function for creating a fully connected neural network.
    This version uses BatchNorm between linear layers.

    :param dims: The size of the layers in the network (at least 2)
    :return: nn.Sequential containing all the layers
    """

    layers = [nn.Linear(dims[0], dims[1])]

    for in_dim, out_dim in zip(dims[1:], dims[2:]):
        layers.append(nn.BatchNorm1d(in_dim))
        layers.append(nn.ReLU())
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
