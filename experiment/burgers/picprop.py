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
from cipinn.pde.burgers import BurgersPDE
from cipinn.data.data import DataGenerator
from cipinn.data.utils import FixedBounds  # get fixed bounds
from cipinn.model.pinn import PINN
from metaoptim.metaoptim import MetaOptimizer
from metaoptim.neumann import NeumannSeries
from metaoptim.cg import CG
from metaoptim.reverse import Reverse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--x_q', type=float, default=0.2)
parser.add_argument('--t_q', type=float, default=0.0)

# for pinn
parser.add_argument('--n_i', type=int, default=256)
parser.add_argument('--sigma_i', type=float, default=0.1) # 0.00
parser.add_argument('--lamb_i', type=float, default=1.)
parser.add_argument('--n_b', type=int, default=200)
parser.add_argument('--sigma_b', type=float, default=0.00)
parser.add_argument('--lamb_b', type=float, default=1.)
parser.add_argument('--n_f', type=int, default=10000)
parser.add_argument('--sigma_f', type=float, default=0.00)
parser.add_argument('--lamb_f', type=float, default=1.)
parser.add_argument('--n_t', type=int, default=25600)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--minibatch', type=bool, default=False)
parser.add_argument('--f_batch_size', type=int, default=100)

# for confidence region
parser.add_argument('--confidence', type=float, default=0.95)
parser.add_argument('--region', type=str, default='fixed')  # e.g. gaussian, chi2, cross_product, fixed

# for meta-optimizer
parser.add_argument('--meta_optim', type=str, default='sgd')
parser.add_argument('--meta_lr', type=float, default=1e-3)
parser.add_argument('--hypergrad', type=str, default='reverse') # e.g. ns, cg, reverse
parser.add_argument('--meta_steps', type=int, default=500)
parser.add_argument('--inner_steps', type=int, default=100)
parser.add_argument('--warmup_steps', type=int, default=20000)

# for implicit differentiation (neumann series, conjugate gradient)
parser.add_argument('--truncate_iter', type=int, default=200)
parser.add_argument('--lamb', type=float, default=0.01)
parser.add_argument('--max_grad_norm', type=float, default=10.)

# for neumann series
parser.add_argument('--ns_alpha', type=float, default=1e-4)

# for reverse
parser.add_argument('--k_history', type=int, default=2000)

parser.add_argument('--device', type=str, default="cuda:0")

args = parser.parse_args()

logger = Logger('bo')
logger.add_params(vars(args))
set_seed(args.seed)

device = torch.device(args.device)

pde = BurgersPDE()
data_generator = DataGenerator(pde, device=device)

x_i, u_i = data_generator.gen_data_i(n=args.n_i, sigma=args.sigma_i, random_loc=False)
u_i.requires_grad_(True)
x_b, u_b = data_generator.gen_data_b(n=args.n_b, sigma=args.sigma_b, random_loc=False)
x_f, f = data_generator.gen_data_f(n=args.n_f, sigma=args.sigma_f)
x_t, u_t, f_t = data_generator.gen_data_t(n=args.n_t)
x_q = torch.from_numpy(np.array([[args.x_q, args.t_q]])).float().to(device)

bounds = FixedBounds(confidence=args.confidence, u_str="u_i", device=device)

if args.minibatch:
    f_dataset = TensorDataset(x_f, f)
    f_dataloader = DataLoader(f_dataset, batch_size=args.f_batch_size, shuffle=True)
else:
    f_dataloader = [None] * args.inner_steps

model = PINN(pde=pde, hidden_size=20, depth=8, lamb_i=args.lamb_i,
             lamb_b=args.lamb_b, lamb_f=args.lamb_f).to(device)
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
    meta_optim=meta_optim_cons([u_i], lr=args.meta_lr),
    hypergrad=hypergrad,
    max_grad_norm=args.max_grad_norm
)

from datetime import datetime
time_filepath = os.path.join(logger.logdir, "timer.csv")

# Find mean values

param_traj = {'low': [], 'high': []}
obj_traj = {'low': [], 'high': []}
for mode in ['low', 'high']:
    start_time = datetime.now()
    dt_string = start_time.strftime("%d/%m/%Y %H:%M:%S")
    with open(time_filepath, 'a') as fwriter:
        fwriter.write(mode + ", " + dt_string + "\n")

    # u_i.data = torch.zeros_like(u_i)
    u_i.data = bounds.mean.data

    optimizer.state = collections.defaultdict(dict)
    metaoptim.state = collections.defaultdict(dict)
    model.randominit()

    def inner_loss_f(_params, _hyperparams):
        data_f = next(iter(f_dataloader))
        data_f = [x_f, f] if data_f is None else data_f
        loss = model.loss(data_i=[x_i, u_i], data_b=[x_b, u_b], data_f=data_f)
        return loss

    def meta_loss_f(_params, _hyperparams):
        return torch.mean(model(x_q)) * (1.0 if mode == 'low' else -1.0)

    def inner_update_opt(_params, _hyperparams):
        _cp = deepcopy(model)
        for p1, p2 in zip(_cp.parameters(), _params):
            p1.data = p2.data
        _loss = _cp.loss(data_i=[x_i, _hyperparams[0]])
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
        # u_i.data = project(u_i.data, mean=torch.zeros_like(u_i), sigma=args.sigma_i,
        #                    confidence=args.confidence, region=args.region, bounds=bounds)
        u_i.data = project(u_i.data, mean=bounds.mean, sigma=args.sigma_i,
                           confidence=args.confidence, region=args.region, bounds=bounds)
        print(u_i.reshape((-1,)))
        logger.commit(epoch=0 if mode == 'low' else 1, step=meta_step)
        param_traj[mode].append(u_i.detach().cpu().numpy())
        obj_traj[mode].append(meta_loss.item() * (1 if mode == 'low' else -1))
    # save results
    pred = model(x_t)
    logger.savez(file='rec-{}-{}-{}.npz'.format(round(args.x_q, 2), round(args.t_q, 2), mode),
                 param_traj=np.array(param_traj[mode]),
                 obj_traj=np.array(obj_traj[mode]),
                 x_t=x_t.detach().cpu().numpy(),
                 pred=pred.detach().cpu().numpy(),
                 )

    # save model
    logger.save_model(file='model-{}-{}-{}.pkl'.format(round(args.x_q, 2), round(args.t_q, 2), mode), model=model)

    end_time = datetime.now()
    dt_string = end_time.strftime("%d/%m/%Y %H:%M:%S")
    with open(time_filepath, 'a') as fwriter:
        fwriter.write(mode + ", " + dt_string + "\n")

    time_delta = (end_time - start_time)
    total_seconds = time_delta.total_seconds()
    minutes_string = str(total_seconds/60)
    with open(time_filepath, 'a') as fwriter:
        fwriter.write(mode + ", " + minutes_string + "\n")

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

data = pde.u.reshape((100, 256)).T

sqrt = int(np.sqrt(args.n_t))
img = ax1.imshow(data, interpolation='nearest', cmap='seismic',
                 extent=[0.0, 1.0, -1.0, 1.0], origin='lower', aspect='auto')
plt.colorbar(img, ax=ax1)
ax1.set_xlabel("$t$", fontsize=16)
ax1.set_ylabel("$x$", fontsize=16)
points = x_b.detach().cpu().numpy()
ax1.scatter(points[:, 1], points[:, 0], marker='x', color='black', label='boundary')
points = x_i.detach().cpu().numpy()
ax1.scatter(points[:, 1], points[:, 0], marker='x', color='grey', label='initial')
ax1.scatter([args.t_q], [args.x_q], marker='x', color='yellow', label='query')
ax1.legend()

ax2.plot(np.arange(len(obj_traj['low'])), obj_traj['low'], label='bo:low', c='r')
ax2.plot(np.arange(len(obj_traj['high'])), obj_traj['high'], label='bo:high', c='g')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(logger.logdir, 'viz.pdf'))
