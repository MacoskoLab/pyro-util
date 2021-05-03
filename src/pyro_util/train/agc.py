import torch
from torch._six import inf
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from typing import Union, Iterable

from pyro.optim import PyroOptim

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def clip_grad_agc_(
    parameters: _tensor_or_tensors, clipping: float, norm_type: float = 2.0
) -> torch.Tensor:
    r"""Adaptive gradient clipping, implemented as a util à la clip_grad_norm_

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clipping (float or int): clipping factor used to compute max_norm
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    clipping = float(clipping)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device

    if norm_type == inf:
        param_norm = max(p.detach().abs().max().to(device) for p in parameters)
        grad_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        param_norm = torch.norm(
            torch.stack(
                [torch.norm(p.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
        grad_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )

    max_norm = param_norm * clipping
    clip_coef = max_norm / (grad_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return grad_norm


def clip_grad_w_agc(params, clip_norm=None, clip_value=None, clip_agc=None):
    """Monkey-patching this ability into Pyro"""
    if clip_norm is not None:
        clip_grad_norm_(params, clip_norm)
    if clip_value is not None:
        clip_grad_value_(params, clip_value)
    if clip_agc is not None:
        clip_grad_agc_(params, clip_agc)


PyroOptim._clip_grad = staticmethod(clip_grad_w_agc)
