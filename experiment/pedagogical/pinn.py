import os
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse

from cipinn.utils.logger import Logger
from cipinn.utils.utils import set_seed
from cipinn.pde.pedagogical import PedagogicalPDE
from cipinn.data.data import DataGenerator
from cipinn.model.pinn import PINN

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_b', type=int, default=2)
parser.add_argument('--sigma_b', type=float, default=0.00)
parser.add_argument('--lamb_b', type=float, default=100.)
parser.add_argument('--n_f', type=int, default=1000)
parser.add_argument('--sigma_f', type=float, default=0.00)
parser.add_argument('--lamb_f', type=float, default=1.)
parser.add_argument('--n_t', type=int, default=101)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=4)
parser.add_argument('--steps_per_epoch', type=int, default=1000)
parser.add_argument('--minibatch', type=bool, default=False)
parser.add_argument('--device', type=str, default="cpu")
args = parser.parse_args()

logger = Logger('pinn')
logger.add_params(vars(args))
set_seed(args.seed)

device = torch.device(args.device)

pde = PedagogicalPDE()
data_generator = DataGenerator(pde, device=device)
model = PINN(pde=pde, lamb_b=args.lamb_b, lamb_f=args.lamb_f).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

x_b, u_b = data_generator.gen_data_b(n=args.n_b, sigma=args.sigma_b)
x_f, f = data_generator.gen_data_f(n=args.n_f, sigma=args.sigma_f)
x_t, u_t, f_t = data_generator.gen_data_t(n=args.n_t)
if args.minibatch:
    f_dataset = TensorDataset(x_f, f)
    f_dataloader = DataLoader(f_dataset, batch_size=args.n_f // args.steps_per_epoch, shuffle=True)
else:
    f_dataloader = [None] * args.steps_per_epoch

step_cnt = 0
for epoch in tqdm(range(args.epoch)):
    # training
    for data_f in f_dataloader:
        data_f = [x_f, f] if data_f is None else data_f
        loss = model.loss(data_b=[x_b, u_b], data_f=data_f)
        logger.add_metric('train_loss', loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_cnt += 1
    # evaluation
    loss_u = model.loss(data_u=[x_t, u_t])
    loss_f = model.loss(data_f=[x_t, f_t])
    logger.add_metric('test_loss_u', loss_u.item() / model.lamb_u)
    logger.add_metric('test_loss_f', loss_f.item() / model.lamb_f)
    logger.commit(epoch=epoch, step=step_cnt)

# plot
pred = model(x_t)
plt.plot(x_t.detach().cpu().numpy().reshape((-1, )),
         u_t.detach().cpu().numpy().reshape((-1, )), c='b', linestyle='--', label=r'Exact')
plt.plot(x_t.detach().cpu().numpy().reshape((-1, )),
         pred.detach().cpu().numpy().reshape((-1, )), c='r', linestyle=':', label=r'Pred')
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(logger.logdir, 'viz.pdf'))
plt.show()

# save results
pred = model(x_t)
logger.savez(file='rec.npz',
             x_b=x_b.detach().cpu().numpy(),
             u_b=u_b.detach().cpu().numpy(),
             x_t=x_t.detach().cpu().numpy(),
             pred=pred.detach().cpu().numpy(),
             )

# save model
logger.save_model(file='model.pkl', model=model)
