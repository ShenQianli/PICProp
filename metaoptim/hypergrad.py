from typing import Callable, Iterable, Union, Sequence
from abc import ABCMeta, abstractmethod
import torch
from torch import Tensor

from metaoptim.utils import grad_unused_zero


class HyperGrad(metaclass=ABCMeta):
    """
    Hyper-Gradient
    """

    @abstractmethod
    def grad(
            self,
            inner_loss_f: Callable[[Sequence[Tensor], Sequence[Tensor]], Tensor],
            meta_loss_f: Callable[[Sequence[Tensor], Sequence[Tensor]], Tensor],
            params: Iterable,
            hyperparams: Iterable,
            **kwargs,
    ) -> Sequence[Tensor]:
        """
        Calculate the hyper-gradient

        :param inner_loss_f:
        :param meta_loss_f:
        :param params:
        :param hyperparams:
        :kwargs:
        :return: hyper-gradient
        """
        pass


class ImplicitDiff(HyperGrad):
    """
    Implicit Differential
    """
    def grad(
            self,
            inner_loss_f: Callable[[Sequence[Tensor], Sequence[Tensor]], Tensor],
            meta_loss_f: Callable[[Sequence[Tensor], Sequence[Tensor]], Tensor],
            params: Sequence[Tensor],
            hyperparams: Sequence[Tensor],
            **kwargs,
    ) -> Sequence[Tensor]:
        inner_loss = inner_loss_f(params, hyperparams)
        meta_loss = meta_loss_f(params, hyperparams)
        d_inner_loss_d_params = grad_unused_zero(inner_loss, params, create_graph=True)
        d_meta_loss_d_params = grad_unused_zero(meta_loss, params)
        inverse_hvp = self._inverse_hessian_vector_product(
            grads=d_inner_loss_d_params,
            params=params,
            v=d_meta_loss_d_params
        )
        hypergrads = grad_unused_zero(d_inner_loss_d_params, hyperparams, grad_outputs=inverse_hvp,)

        return [-hg for hg in hypergrads]

    def _inverse_hessian_vector_product(
            self,
            grads: Sequence[Tensor],
            params: Sequence[Tensor],
            v: Sequence[Tensor],
    ) -> Sequence[Tensor]:
        return v
