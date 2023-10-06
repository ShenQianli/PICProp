from typing import Sequence
import torch
from torch import Tensor


from metaoptim.hypergrad import ImplicitDiff
from metaoptim.utils import gather_flat_vec


class NeumannSeries(ImplicitDiff):
    """
    Implicit differential with Neumann Series
    """
    def __init__(
            self,
            alpha: float = 1e-3,
            truncate_iter: int = 100,
            tolerance: float = 1e-6,
            lamb: float = 0.0
    ):
        self._alpha = alpha
        self._truncated_iter = truncate_iter
        self._tolerence = tolerance
        self._lamb = lamb

    def _inverse_hessian_vector_product(
            self,
            grads: Sequence[Tensor],
            params: Sequence[Tensor],
            v: Sequence[Tensor],
    ) -> Sequence[Tensor]:
        s = v
        debug = []
        for i in range(self._truncated_iter):
            hvp = torch.autograd.grad(
                    grads,
                    params,
                    grad_outputs=v,
                    retain_graph=True,
                    allow_unused=True,
            )
            v = [(1 - self._lamb * self._alpha) * vv - self._alpha * pp for (vv, pp) in zip(v, hvp)]
            s = [ss + vv for (ss, vv) in zip(s, v)]
            debug.append(gather_flat_vec(v).abs().max().item())
            if gather_flat_vec(v).abs().max() <= self._tolerence:
                break
        # print(debug)
        return [self._alpha * ss for ss in s]



