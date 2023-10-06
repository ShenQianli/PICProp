from typing import Sequence
import torch
from torch import Tensor


from metaoptim.hypergrad import ImplicitDiff


class CG(ImplicitDiff):
    """
    Implicit differential with Conjugate Gradient
    """
    def __init__(
            self,
            truncate_iter: int = 100,
            tolerance: float = 1e-6,
            lamb: float = 0.0
    ):
        self._truncated_iter = truncate_iter
        self._tolerence = tolerance
        self._lamb = lamb

    def _inverse_hessian_vector_product(
            self,
            grads: Sequence[Tensor],
            params: Sequence[Tensor],
            v: Sequence[Tensor],
    ) -> Sequence[Tensor]:
        p = r = v
        x = [torch.zeros_like(pp) for pp in p]
        rr = list_dot(r, r)
        r0 = list_dot(r, r)
        debug = []
        for idx in range(self._truncated_iter):
            pa = torch.autograd.grad(
                grads,
                params,
                grad_outputs=p,
                retain_graph=True,
                allow_unused=True
            )
            pa = [curr_pa + self._lamb * curr_p for (curr_pa, curr_p) in zip(pa, p)]
            pap = list_dot(pa, p)
            # a = rr / (pap + 1e-6)
            a = rr / pap
            x = [curr_x + a * curr_p for (curr_x, curr_p) in zip(x, p)]
            r = [curr_r - a * curr_pa for (curr_r, curr_pa) in zip(r, pa)]
            rr_new = list_dot(r, r)
            # b = rr_new / (rr + 1e-6)
            b = rr_new / rr
            rr = rr_new
            debug.append(rr)
            if rr / r0 < self._tolerence:
                # print(idx)
                break
            p = [curr_r + b * curr_p for (curr_r, curr_p) in zip(r, p)]

        # print(debug[:10])
        # print(debug[-10:])
        return x


def list_dot(l1, l2):
    assert len(l1) == len(l2)
    total = 0
    for i, j in zip(l1, l2):
        if i is None or j is None:
            continue
        assert i.shape == j.shape
        if len(i.shape) > 1:
            for ii, jj in zip(i, j):
                assert ii.shape == jj.shape
                total += torch.dot(ii, jj)
        else:
            total += torch.dot(i, j)
    return total

