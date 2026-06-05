import torch
import torch.nn as nn

_LIGER_FN = None

try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    _LIGER_FN = LigerRMSNormFunction
except ImportError:
    pass


class RMSNorm(nn.Module):
    """RMSNorm avec fusion Triton optionnelle via liger-kernel.

    Le poids est TOUJOURS enregistré sur ce module (`self.weight`),
    que Liger soit installé ou non — garantit la cohérence du state_dict
    entre runs avec/sans Liger et évite tout AttributeError sur .weight.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _LIGER_FN is not None:
            return _LIGER_FN.apply(x, self.weight, self.eps)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
