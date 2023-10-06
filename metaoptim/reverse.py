from typing import Callable, Iterable, Union, Sequence
import torch
from torch import Tensor

from metaoptim.hypergrad import HyperGrad
from metaoptim.utils import grad_unused_zero


class Reverse(HyperGrad):
    """
    Reverse-mode differentiation
    https://github.com/JunjieYang97/stocBiO/blob/16ef25da262f48553d2eac63662394f40a1db071/Hyperparameter-optimization/hypergrad/hypergradients.py
    """

    def grad(
            self,
            inner_loss_f: Callable[[Sequence[Tensor], Sequence[Tensor]], Tensor],
            meta_loss_f: Callable[[Sequence[Tensor], Sequence[Tensor]], Tensor],
            params: Sequence[Tensor],
            hyperparams: Sequence[Tensor],
            **kwargs,
    ) -> Sequence[Tensor]:
        params_history = kwargs.get('params_history')
        assert params_history is not None
        inner_update_opt = kwargs.get('inner_update_opt')
        assert inner_update_opt is not None

        params_history = [[w.detach().requires_grad_(True) for w in p] for p in params_history]
        meta_loss = meta_loss_f(params, hyperparams)
        d_meta_loss_d_params = grad_unused_zero(meta_loss, params)
        grads = [torch.zeros_like(hp) for hp in hyperparams]
        K = len(params_history) - 1
        for k in range(-2, -(K + 2), -1):
            w_mapped = inner_update_opt(params_history[k], hyperparams)
            bs = grad_unused_zero(w_mapped, hyperparams, grad_outputs=d_meta_loss_d_params)
            grads = [g + b for g, b in zip(grads, bs)]
            d_meta_loss_d_params = grad_unused_zero(w_mapped, params_history[k], grad_outputs=d_meta_loss_d_params)

        return grads
