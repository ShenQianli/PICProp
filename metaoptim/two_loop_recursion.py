from typing import Sequence, Union
from collections import defaultdict

import torch
from torch import Tensor


from .hypergrad import ImplicitDiff
from .utils import gather_flat_vec, parse_flat_grad


class TwoLoopRecursion(ImplicitDiff):
    """
    Implicit differential
    Approximate inverse hessian vector product with two-loop recursion
    """
    def __init__(self):
        self._state = defaultdict()

    def _inverse_hessian_vector_product(
            self,
            grads: Sequence[Tensor],
            params: Sequence[Tensor],
            v: Sequence[Tensor],
    ) -> Sequence[Tensor]:
        state = self._state
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')

        num_old = len(old_dirs)

        al = state['al']

        # iteration in L-BFGS loop collapsed to use just one buffer
        q = gather_flat_vec(v)
        for i in range(num_old - 1, -1, -1):
            al[i] = old_stps[i].dot(q) * ro[i]
            q.add_(-al[i], old_dirs[i])

        # multiply by initial Hessian
        # r is the final direction
        r = torch.mul(q, H_diag)
        for i in range(num_old):
            be_i = old_dirs[i].dot(r) * ro[i]
            r.add_(al[i] - be_i, old_stps[i])

        return parse_flat_grad(params, r)

    def update_state(
            self,
            state: defaultdict
    ):
        self._state = state
