"""
A simple meta-learning example

max_x u^T y(x)

s.t. y(x) = argmin_y y^Ty - 2 x^T y, x^Tx <= 1

The oracle solution should be x* = y* = u / u^Tu
"""

import torch
from torch import optim
import argparse
from copy import deepcopy

from metaoptim.metaoptim import MetaOptimizer
from metaoptim.neumann import NeumannSeries
from metaoptim.cg import CG
from metaoptim.reverse import Reverse

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-2)

# for meta-optimizer
parser.add_argument('--meta_lr', type=float, default=1e-2)
parser.add_argument('--hypergrad', type=str, default='reverse')
parser.add_argument('--meta_steps', type=int, default=50)
parser.add_argument('--inner_steps', type=int, default=100)

# for implicit differentiation (neumann series, conjugate gradient)
parser.add_argument('--truncate_iter', type=int, default=200)
parser.add_argument('--lamb', type=float, default=0.00)
parser.add_argument('--max_grad_norm', type=float, default=10.)

# for neumann series
parser.add_argument('--ns_alpha', type=float, default=1e-4)

# for reverse
parser.add_argument('--k_history', type=int, default=100)

args = parser.parse_args()


u = torch.tensor([3, 4]).float()
x = torch.tensor([0, 0]).float().requires_grad_(True)
y = torch.zeros_like(u).requires_grad_(True)

optimizer = optim.SGD([y], lr=args.lr)

if args.hypergrad == 'ns':
    hypergrad = NeumannSeries(
        alpha=args.ns_alpha,
        truncate_iter=args.truncate_iter,
        lamb=args.lamb,
    )
elif args.hypergrad == 'cg':
    hypergrad = CG(
        truncate_iter=args.truncate_iter,
        lamb=args.lamb,
    )
elif args.hypergrad == 'reverse':
    hypergrad = Reverse()
else:
    raise ValueError('Unknown hypergrad: {}'.format(args.hypergrad))

metaoptim = MetaOptimizer(
    inner_optim=optimizer,
    meta_optim=torch.optim.SGD([x], lr=args.meta_lr),
    hypergrad=hypergrad,
    max_grad_norm=args.max_grad_norm
)


def inner_loss_f(_params, _hyperparams):
    _x = _hyperparams[0]
    _y = _params[0]
    return torch.sum(_y * _y) - 2 * torch.sum(_x * _y)


def meta_loss_f(_params, _hyperparams):
    _y = _params[0]
    return - torch.sum(u * _y)


def inner_update_opt(_params, _hyperparams):
    _cp = deepcopy(_params)
    _loss = torch.sum(_cp[0] * _cp[0]) - 2 * torch.sum(_hyperparams[0] * _cp[0])
    _grads = torch.autograd.grad(_loss, _cp, create_graph=True)
    _updated = [_p - args.lr * _g for _p, _g in zip(_params, _grads)]
    return _updated


for meta_step in range(args.meta_steps):
    inner_loss, meta_loss = metaoptim.step(
        n_inner_steps=args.inner_steps,
        inner_loss_f=inner_loss_f,
        meta_loss_f=meta_loss_f,
        k_history=args.k_history,
        inner_update_opt=inner_update_opt,
    )
    k = 1 / torch.sqrt(torch.sum(x.data * x.data)).item()
    if k < 1:
        x.data = x.data / torch.sqrt(torch.sum(x.data * x.data))
    print('step {}: x = {}, y = {}'.format(meta_step, x, y))
