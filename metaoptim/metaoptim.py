from typing import Callable, List, Optional, Tuple, Sequence
from abc import ABCMeta, abstractmethod
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from copy import deepcopy

from .hypergrad import HyperGrad
from .utils import gather_flat_grad, assign_grad


class MetaOptimizer(metaclass=ABCMeta):
    """
    Meta Optimizer Base
    """

    inner_optim: Optimizer
    meta_optim: Optimizer
    hyper_grad: HyperGrad
    _params: List
    _hyperparams: List

    def __init__(
            self,
            inner_optim: Optimizer,
            meta_optim: Optimizer,
            hypergrad: HyperGrad,
            max_grad_norm: float = 10.0,
    ) -> None:
        self.inner_optim = inner_optim
        self.meta_optim = meta_optim
        self.hypergrad = hypergrad
        self.max_grad_norm = max_grad_norm

        # only single parameter group supported
        if len(self.inner_optim.param_groups) != 1 or \
                len(self.meta_optim.param_groups) != 1:
            raise ValueError("MetaOptimizer doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.inner_optim.param_groups[0]['params']
        self._hyperparams = self.meta_optim.param_groups[0]['params']

    def step(
            self,
            n_inner_steps: int,
            inner_loss_f: Callable[[Sequence[Tensor], Sequence[Tensor]], Tensor],
            meta_loss_f: Callable[[Sequence[Tensor], Sequence[Tensor]], Tensor],
            k_history: int = 0,
            inner_update_opt: Optional[Callable] = None,
            inner_retain_graph: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        inner_step_cnt = 0
        params_history = []
        # inner optimization
        while inner_step_cnt < n_inner_steps:
            def _inner_closure():
                self.inner_optim.zero_grad()
                inner_loss = inner_loss_f(self._params, self._hyperparams)
                inner_loss.backward(retain_graph=inner_retain_graph)

            self.inner_step(_inner_closure)
            inner_step_cnt += 1
            if inner_step_cnt > n_inner_steps - k_history:
                params_history.append([deepcopy(p) for p in self._params])

        # meta optimization
        kwargs = {}
        if k_history > 0:
            kwargs['params_history'] = params_history
        if inner_update_opt is not None:
            kwargs['inner_update_opt'] = inner_update_opt
        inner_loss, meta_loss = self.meta_step(inner_loss_f, meta_loss_f, **kwargs)

        return inner_loss, meta_loss

    def inner_step(
            self,
            closure: Optional[Callable] = None
    ):
        self.inner_optim.step(closure=closure)

    def compute_grad(
            self,
            inner_loss_f: Callable[[Sequence[Tensor], Sequence[Tensor]], Tensor],
            meta_loss_f: Callable[[Sequence[Tensor], Sequence[Tensor]], Tensor],
            **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """

        :param inner_loss_f:
        :param meta_loss_f:
        :kwargs
        :return:
        """
        self.meta_optim.zero_grad()
        inner_loss = inner_loss_f(self._params, self._hyperparams)
        meta_loss = meta_loss_f(self._params, self._hyperparams)
        hypergrads = self.hypergrad.grad(
            inner_loss_f=inner_loss_f,
            meta_loss_f=meta_loss_f,
            params=self._params,
            hyperparams=self._hyperparams,
            **kwargs,
        )
        grads = torch.autograd.grad(
            meta_loss,
            self._hyperparams,
            grad_outputs=torch.ones_like(meta_loss),
            allow_unused=True,
        )
        flat_hypergrads = gather_flat_grad(params=self._hyperparams, grads=hypergrads)
        flat_grads = gather_flat_grad(params=self._hyperparams, grads=grads)
        update = flat_hypergrads + flat_grads
        assign_grad(self._hyperparams, update)

        return inner_loss, meta_loss

    def meta_step(
            self,
            inner_loss_f: Callable[[Sequence[Tensor], Sequence[Tensor]], Tensor],
            meta_loss_f: Callable[[Sequence[Tensor], Sequence[Tensor]], Tensor],
            **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Take a meta-step.
        Optimizers that require `closure` for updating are not supported.
        You need to define a `closure` function by yourself. (see `MetaOptimizer.compute_grad`\
        for more info)

        :param inner_loss_f:
        :param meta_loss_f:
        :kwargs:
        :return:
        """
        for p in self._hyperparams:
            p.grad = None
        inner_loss, meta_loss = self.compute_grad(inner_loss_f, meta_loss_f, **kwargs)

        # grad clipping
        if self.max_grad_norm is not None:
            clip_grad_norm_(self._hyperparams, max_norm=self.max_grad_norm)

        self.meta_optim.step()

        return inner_loss, meta_loss
