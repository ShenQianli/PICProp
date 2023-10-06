import os
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
import argparse
import numpy as np
import collections
from copy import deepcopy

from cipinn.utils.logger import Logger
from cipinn.utils.utils import set_seed, project
from cipinn.pde.pedagogical import PedagogicalPDE
from cipinn.data.data import DataGenerator
from cipinn.model.pinn import PINN
from metaoptim.metaoptim import MetaOptimizer
from metaoptim.neumann import NeumannSeries
from metaoptim.cg import CG
from metaoptim.reverse import Reverse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--x_q', type=float, default=0.2)

# for pinn
parser.add_argument('--n_b', type=int, default=2)
parser.add_argument('--sigma_b', type=float, default=0.05)
parser.add_argument('--lamb_b', type=float, default=100.)
parser.add_argument('--n_f', type=int, default=1000)
parser.add_argument('--sigma_f', type=float, default=0.00)
parser.add_argument('--lamb_f', type=float, default=1.)
parser.add_argument('--n_t', type=int, default=101)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--minibatch', type=bool, default=False)
parser.add_argument('--f_batch_size', type=int, default=100)

# for confidence region
parser.add_argument('--confidence', type=float, default=0.95)
parser.add_argument('--region', type=str, default='chi2')

# for meta-optimizer
parser.add_argument('--meta_optim', type=str, default='sgd')
parser.add_argument('--meta_lr', type=float, default=1e-2)
parser.add_argument('--hypergrad', type=str, default='ns')
parser.add_argument('--meta_steps', type=int, default=50)
parser.add_argument('--inner_steps', type=int, default=500)
parser.add_argument('--warmup_steps', type=int, default=2000)

# for implicit differentiation (neumann series, conjugate gradient)
parser.add_argument('--truncate_iter', type=int, default=200)
parser.add_argument('--lamb', type=float, default=.01)
parser.add_argument('--max_grad_norm', type=float, default=10.)

# for neumann series
parser.add_argument('--ns_alpha', type=float, default=1e-4)

# for reverse
parser.add_argument('--k_history', type=int, default=2000)

parser.add_argument('--device', type=str, default="cpu")

args = parser.parse_args()

logger = Logger('{}/picprop_{}'.format(args.region, round(args.x_q, 2)), with_timestamp=False)
logger.add_params(vars(args))
set_seed(args.seed)

device = torch.device(args.device)

pde = PedagogicalPDE()
data_generator = DataGenerator(pde, device=device)

x_b, u_b = data_generator.gen_data_b(n=args.n_b, sigma=args.sigma_b)
# load randomly sampled boundary data. See ./data/gen_data.py for details.
if args.region == 'chi2':
    # 1 point from gaussian for chi2 CI on boundary
    z = np.load('data/chi2.npz')['z']
    z_bar = torch.from_numpy(z[0]).reshape((-1, 1)).to(device)
elif args.region == 't2':
    # 5 points from gaussian for t2 CI on boundary
    z = np.load('data/t2.npz')['z']
    n = len(z)
    z_bar = torch.from_numpy(np.mean(z, 0)).reshape((-1, 1)).to(device)
    S = torch.from_numpy(np.cov(z, rowvar=False)).to(device)
elif args.region == 'hoeffding':
    # 5 points from uniform for hoeffding CI on boundary
    rec = np.load('data/hoeffding.npz')
    lb, ub = rec['bound'][:, 0:1], rec['bound'][:, 1:2]
    lb, ub = torch.from_numpy(lb).to(device), torch.from_numpy(ub).to(device)
    z_bar = torch.from_numpy(np.zeros((2, 1))).to(device) # no use
else:
    raise NotImplementedError()
u_b.requires_grad_(True)
x_f, f = data_generator.gen_data_f(n=args.n_f, sigma=args.sigma_f)
x_t, u_t, f_t = data_generator.gen_data_t(n=args.n_t)
x_q = torch.from_numpy(np.array([[args.x_q]])).float().to(device)
if args.minibatch:
    f_dataset = TensorDataset(x_f, f)
    f_dataloader = DataLoader(f_dataset, batch_size=args.f_batch_size, shuffle=True)
else:
    f_dataloader = [None] * args.inner_steps

model = PINN(pde=pde, lamb_b=args.lamb_b, lamb_f=args.lamb_f).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

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

if args.meta_optim == 'sgd':
    meta_optim_cons = torch.optim.SGD
elif args.meta_optim == 'adam':
    meta_optim_cons = torch.optim.Adam
else:
    raise ValueError('Unknown meta_optim: {}'.format(args.meta_optim))

metaoptim = MetaOptimizer(
    inner_optim=optimizer,
    meta_optim=meta_optim_cons([u_b], lr=args.meta_lr),
    hypergrad=hypergrad,
    max_grad_norm=args.max_grad_norm
)

param_traj = {'low': [], 'high': []}
obj_traj = {'low': [], 'high': []}
for mode in ['low', 'high']:
    u_b.data = deepcopy(z_bar.data)
    optimizer.state = collections.defaultdict(dict)
    metaoptim.state = collections.defaultdict(dict)
    model.randominit()

    def inner_loss_f(_params, _hyperparams):
        data_f = next(iter(f_dataloader))
        data_f = [x_f, f] if data_f is None else data_f
        loss = model.loss(data_b=[x_b, u_b], data_f=data_f)
        return loss

    def meta_loss_f(_params, _hyperparams):
        return torch.mean(model(x_q)) * (1.0 if mode == 'low' else -1.0)

    def inner_update_opt(_params, _hyperparams):
        _cp = deepcopy(model)
        for p1, p2 in zip(_cp.parameters(), _params):
            p1.data = p2.data
        _loss = _cp.loss(data_b=[x_b, _hyperparams[0]])
        _grads = torch.autograd.grad(_loss, list(_cp.parameters()), create_graph=True)
        _updated = [_p - 1e-1 * _g for _p, _g in zip(_params, _grads)]
        return _updated

    for meta_step in range(args.meta_steps):
        n_inner_steps = max(args.warmup_steps, args.inner_steps) if meta_step == 0 else args.inner_steps
        inner_loss, meta_loss = metaoptim.step(
            n_inner_steps=n_inner_steps,
            inner_loss_f=inner_loss_f,
            meta_loss_f=meta_loss_f,
            k_history=args.k_history,
            inner_update_opt=inner_update_opt,
        )
        logger.add_metric('inner_loss', inner_loss.item())
        logger.add_metric('meta_loss', meta_loss.item())
        kwargs = {}
        if args.region == 't2':
            kwargs['n'] = n
            kwargs['S'] = S
        if args.region == 'hoeffding':
            u_b.data = torch.max(torch.min(u_b.data, ub), lb)
        else:
            u_b.data = project(u_b.data, mean=z_bar, sigma=args.sigma_b,
                               confidence=args.confidence, region=args.region, **kwargs)

        logger.commit(epoch=0 if mode == 'low' else 1, step=meta_step)
        param_traj[mode].append(u_b.detach().cpu().numpy())
        obj_traj[mode].append(meta_loss.item() * (1 if mode == 'low' else -1))
    # save results
    pred = model(x_t)
    logger.savez(file='rec-{}-{}.npz'.format(round(args.x_q, 2), mode),
                 param_traj=np.array(param_traj[mode]),
                 obj_traj=np.array(obj_traj[mode]),
                 x_t=x_t.detach().cpu().numpy(),
                 pred=pred.detach().cpu().numpy(),
                 )

    # save model
    # logger.save_model(file='model-{}-{}.pkl'.format(round(args.x_q, 2), mode), model=model)

# plot
plt.plot(np.arange(len(obj_traj['low'])), obj_traj['low'], label='bo:low', c='r')
plt.plot(np.arange(len(obj_traj['high'])), obj_traj['high'], label='bo:high', c='g')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(logger.logdir, 'viz.pdf'))
# plt.show()
