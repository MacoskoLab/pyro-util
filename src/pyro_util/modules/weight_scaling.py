import torch
from torch import nn
from torch.functional import F


# gamma-scaled ReLU activation function
class GammaReLU(nn.ReLU):
    # magic number (gamma) for ReLU activation
    RELU_G = 1.7139588594436646

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x, inplace=self.inplace).mul_(GammaReLU.RELU_G)


class WSLinear(nn.Linear):
    """A fully connected linear layer with scaled weight standardization.

    Adapted from https://github.com/vballoli/nfnets-pytorch/blob/main/nfnets/base.py
    Reference: https://github.com/deepmind/deepmind-research/blob/master/nfnets/base.py#L121
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)

        nn.init.kaiming_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.weight.size()[0], requires_grad=True))

    def standardize_weight(self, eps=1e-6):
        mean = torch.mean(self.weight, dim=(1,), keepdim=True)
        var = torch.var(self.weight, dim=(1,), keepdim=True)

        scale = torch.rsqrt(
            torch.max(var * self.in_features, torch.tensor(eps).to(var.device))
        ) * self.gain.view_as(var).to(var.device)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, x: torch.Tensor, eps=1e-6):
        weight = self.standardize_weight(eps)
        return F.linear(x, weight, self.bias)
