from typing import Optional, Iterable, Sequence, Union
import torch
from torch import Tensor


def gather_flat_grad(
        params: Sequence[Tensor],
        grads: Optional[Sequence[Tensor]] = None
) -> Tensor:
    """
    :param params: List of parameters
    :param grads: List of parameters (optional)
    :return: flatten gradients
    """
    views = []
    if grads is None:
        for p in params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
    else:
        for p, g in zip(params, grads):
            if g is None:
                view = p.new(p.numel()).zero_()
            elif g.is_sparse:
                view = g.to_dense().view(-1)
            else:
                view = g.view(-1)
            views.append(view)
    return torch.cat(views, 0)


def gather_flat_vec(
        vec: Sequence[Tensor],
) -> Tensor:
    """
    :param vec: List of parameters
    :return: flatten vector
    """
    return torch.cat([v.view(-1) for v in vec], 0)


def parse_flat_grad(
        params: Sequence[Tensor],
        flat_grads: Tensor
) -> Sequence[Tensor]:
    """

    :param params:
    :param flat_grads:
    :return:
    """
    grad = []
    offset = 0
    for p in params:
        numel = p.numel()
        grad.append(flat_grads[offset:offset + numel].view_as(p.data))
        offset += numel
    assert offset == len(flat_grads)
    return grad


def assign_grad(
        params: Sequence[Tensor],
        flat_grads: Tensor,
):
    """
    :param params: List of parameters
    :param flat_grads: flatten gradients
    :return: None
    """
    offset = 0
    for p in params:
        numel = p.numel()
        p.grad = flat_grads[offset:offset + numel].view_as(p.data)
        offset += numel
    assert offset == len(flat_grads)


def grad_unused_zero(
        outputs: Union[Tensor, Sequence[Tensor]],
        inputs: Sequence[Tensor],
        grad_outputs: Optional[Sequence[Tensor]] = None,
        create_graph: bool = False,
):
    g = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=grad_outputs,
        allow_unused=True,
        create_graph=create_graph,
        retain_graph=create_graph
    )

    g = list(g)
    for i, p, gg in zip(range(len(inputs)), inputs, g):
        if gg is None:
            g[i] = torch.zeros(p.shape)

    return g
