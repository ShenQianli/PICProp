import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from cipinn.model.model import FcNet
from cipinn.utils.logger import Logger
from cipinn.utils.utils import set_seed

plt.rcParams.update({'font.size': 16})
colors = [mcolor.TABLEAU_COLORS[k] for k in mcolor.TABLEAU_COLORS.keys()]

parser = argparse.ArgumentParser()
parser.add_argument('--region', type=str, default='chi2')
parser.add_argument('--lamb', type=float, default=1.)
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

x_qs = [round(i, 2) for i in np.linspace(-1, 1, 6)]

for x_q in x_qs:
    if not os.path.exists('logs/{}/picprop_{}'.format(args.region, x_q)):
        os.system('python picprop.py --region {} --x_q {}'.format(args.region, x_q))

# collect data
data1, data2 = [], []
for x_q in x_qs:
    idx = min(int((x_q + 1) * 50), 99)
    for mode in ['low', 'high']:
        rec = np.load('logs/{}/picprop_{}/recs/rec-{}-{}.npz'.format(args.region, x_q, x_q, mode))
        x_t, pred = rec['x_t'][idx][0], rec['pred'][idx][0]
        data1.append(np.array([x_q, x_q, (1. if mode == 'high' else -1.), pred]))
        x_t, pred = rec['x_t'].reshape((-1, 1)), rec['pred'].reshape((-1, 1))
        data2.append(np.concatenate(
            [np.ones((len(x_t), 1)) * x_q, x_t, np.ones((len(x_t), 1)) * (1. if mode == 'high' else -1.), pred], 1))
data1, data2 = np.array(data1), np.array(data2)
data2 = data2.reshape(-1, data2.shape[-1])

# effipicprop
logger = Logger('14all_{}_lambda={}'.format(args.region, round(args.lamb, 2)), with_timestamp=False)
logger.add_params(vars(args))
set_seed(args.seed)
device = torch.device(args.device)

x1, y1 = torch.from_numpy(data1[:, :-1]).float().to(device), torch.from_numpy(data1[:, -1:]).float().to(device)
x2, y2 = torch.from_numpy(data2[:, :-1]).float(), torch.from_numpy(data2[:, -1:]).float()
dataset = TensorDataset(x2, y2)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

model = FcNet(db=32, depth=2, dx=x2.shape[-1], dy=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

x_t = np.linspace(-1, 1, 101).reshape((-1, 1))
x_t_low = torch.from_numpy(np.concatenate([x_t, x_t, [[-1.]] * len(x_t)], 1)).float().to(device)
x_t_high = torch.from_numpy(np.concatenate([x_t, x_t, [[1.]] * len(x_t)], 1)).float().to(device)

step_cnt = 0
for epoch in tqdm(range(args.epoch)):
    for x, y in dataloader:
        optimizer.zero_grad()
        loss1 = torch.mean((model(x1) - y1) ** 2)
        loss2 = torch.mean((model(x) - y) ** 2)
        loss = (1 - args.lamb) * loss1 + args.lamb * loss2
        logger.add_metric('train_loss', loss.item())
        loss.backward()
        optimizer.step()
        step_cnt += 1
    logger.commit(epoch=epoch, step=step_cnt)

low = model(x_t_low).detach().cpu().numpy()
high = model(x_t_high).detach().cpu().numpy()
logger.savez('ci.npz', high=high, low=low)

# plot
x_t = np.linspace(-1, 1, 101).reshape((-1, 1))
gt = np.sin(np.pi * x_t)
plt.plot(x_t, gt, color='black', label='$\sin(\pi x)$')
x_qs = np.linspace(-1, 1, 41)
plt.plot(x_t, low, color=colors[0], linestyle='--')
plt.plot(x_t, high, color=colors[0], linestyle='--', label=r'EffiPICProp, $\lambda={}$'.format(args.lamb))
z = np.load('data/{}.npz'.format(args.region))['z']
plt.scatter([-1., 1.] * len(z), z.reshape((-1, 1)), marker='x', color='black', label='sample', s=19)

plt.xlabel('x')
plt.ylabel('u')
plt.title('{}'.format(args.region))
plt.legend()
plt.tight_layout()
plt.savefig('plots/effipicprop_{}_lambda={}.pdf'.format(args.region, round(args.lamb, 2)))

