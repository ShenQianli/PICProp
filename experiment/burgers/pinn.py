import sys

import matplotlib as mpl

mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from utils.plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

##
import os
import argparse
from itertools import count
import math

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


##

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)


def none_or_float(value):
    if value == 'None':
        return None
    return float(value)


def none_or_str(value):
    if value == 'None':
        return None
    return value


########== parser start
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max-iter', type=int, default=20)
parser.add_argument('--line-search-fn', type=none_or_str, default='strong_wolfe')
parser.add_argument('--history-size', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=100000)
parser.add_argument('--epoch', type=int, default=100)
args = parser.parse_args()
#############################

#########
result_path = './results/'
if not os.path.isdir(result_path + 'txt/'):
    os.makedirs(result_path + 'txt/')
if not os.path.isdir(result_path + 'npy/'):
    os.makedirs(result_path + 'npy/')
if not os.path.isdir(result_path + 'plot/'):
    os.makedirs(result_path + 'plot/')
result_file_name = str(args.lr) + '_' + str(args.max_iter) + '_' + str(args.line_search_fn)
result_file_name += '_' + str(args.history_size) + '_' + str(args.batch_size)
result_file_name += '_' + str(args.epoch)
filep = open(result_path + 'txt/' + result_file_name + '.txt', 'w')


def output_write(out_str, is_end=True):
    print(out_str, end=' ')
    filep.write(out_str + ' ')
    if is_end == True:
        print('')
        filep.write('\n')


#########

np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.enabled = True


# the fully connected neural network with input dx, output dy, depth hidden layers with dimension db
class FcNet(nn.Module):
    def __init__(self, db=20, depth=8, dx=2, dy=1):
        super(FcNet, self).__init__()
        self.depth = depth
        self.db = db
        fc = []
        for i in range(depth + 1):
            if i == 0:
                fc.append(nn.Linear(dx, db))
            elif i == depth:
                fc.append(nn.Linear(db, dy))
            else:
                fc.append(nn.Linear(db, db))
        self.fc = nn.ModuleList(fc)
        self.randominit()

    def forward(self, z):
        z = self.fc[0](z)
        for i in range(1, self.depth):
            z = torch.tanh(self.fc[i](z))
        return self.fc[self.depth](z)

    def randominit(self):
        for i in range(self.depth + 1):
            out_dim, in_dim = self.fc[i].weight.shape
            xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
            self.fc[i].weight.data.normal_(0, xavier_stddev)
            self.fc[i].bias.data.fill_(0.0)


class ResNet(nn.Module):
    def __init__(self, db=1000, depth=8, dx=2, dy=1):
        super(ResNet, self).__init__()
        self.depth = depth
        self.db = db
        fc = []
        for i in range(depth + 1):
            if i == 0:
                fc.append(nn.Linear(dx, db))
            elif i == depth:
                fc.append(nn.Linear(db, dy))
            else:
                fc.append(nn.Linear(db, db))
        self.fc = nn.ModuleList(fc)
        self.norm = nn.LayerNorm(db)
        # self.randominit()

    def forward(self, z):
        z = self.fc[0](z)
        z_1 = z
        for i in range(1, self.depth):
            z = torch.tanh(self.fc[i](z + z_1))
            # # z = F.softplus(self.fc[i](z + z_1))
            z = self.norm(z)
        return self.fc[self.depth](z + z_1)


model = FcNet().cuda()

# model = ResNet().cuda()

# ===============================================================
# === Burgers eq setup
# ===============================================================

nu = 0.01 / np.pi
noise = 0.0

N_u = 100
N_f = 10000

data = scipy.io.loadmat('./data/burgers_shock.mat')

t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)

xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
uu1 = Exact[0:1, :].T
xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
uu2 = Exact[:, 0:1]
xx3 = np.hstack((X[:, -1:], T[:, -1:]))
uu3 = Exact[:, -1:]

X_u_train = np.vstack([xx1, xx2, xx3])
print(X_u_train.shape)
X_f_train = lb + (ub - lb) * lhs(2, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))
u_train = np.vstack([uu1, uu2, uu3])
print(u_train.shape)

idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx, :]

##############################################
lb = torch.from_numpy(lb).type(torch.FloatTensor).cuda()
ub = torch.from_numpy(ub).type(torch.FloatTensor).cuda()


def net_in(x_in, t_in):
    dum = torch.cat((x_in, t_in), 1)
    return 2.0 * (dum - lb) / (ub - lb) - 1.0


def net_f(x_in, t_in):
    u = model(net_in(x_in, t_in))
    u_t = torch.autograd.grad(u, t_in, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x_in, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_in, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    return u_t + u * u_x - nu * u_xx


x_u = torch.from_numpy(X_u_train[:, 0:1]).type(torch.FloatTensor).cuda()
t_u = torch.from_numpy(X_u_train[:, 1:2]).type(torch.FloatTensor).cuda()

u_data = torch.from_numpy(u_train).type(torch.FloatTensor).cuda()

x_f_all = Variable(torch.from_numpy(X_f_train[:, 0:1]).type(torch.FloatTensor), requires_grad=True).cuda()
t_f_all = Variable(torch.from_numpy(X_f_train[:, 1:2]).type(torch.FloatTensor), requires_grad=True).cuda()

f_data = torch.utils.data.TensorDataset(x_f_all, t_f_all)
f_data_loader = torch.utils.data.DataLoader(f_data, batch_size=args.batch_size, shuffle=True)

##############################################


# ===============================================================
# === training
# ===============================================================
optimizer = optim.LBFGS(model.parameters(), lr=args.lr, max_iter=args.max_iter, line_search_fn=args.line_search_fn,
                        history_size=args.history_size)


def train(is_train=True):
    model.train()
    total_loss = 0
    total_size = 0
    for _, (x_f, t_f) in enumerate(f_data_loader):
        def comp_loss():
            u_pred = model(net_in(x_u, t_u))
            loss = torch.mean((u_pred - u_data) ** 2)
            loss = loss + torch.mean((net_f(x_f, t_f)) ** 2)
            return loss

        def closure():
            optimizer.zero_grad()
            u_pred = model(net_in(x_u, t_u))
            loss = torch.mean((u_pred - u_data) ** 2)
            loss = loss + torch.mean((net_f(x_f, t_f)) ** 2)
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)
        loss = comp_loss()
        total_loss += loss * x_f.size(0)
        total_size += x_f.size(0)
        torch.cuda.empty_cache()
    optimizer.zero_grad()
    total_loss /= total_size
    return total_loss


# == for plot
pl_result = np.zeros((args.epoch + 1, 2, 1))  # epoch * (train, time)
# == main loop start
time_start = time.time()
for epoch in range(0, args.epoch + 1):
    pl_result[epoch, 0, 0] = train(is_train=True)
    pl_result[epoch, 1, 0] = time.time() - time_start
    out_str = '{:d} loss={:.4e} time={:.2f}'.format(epoch, pl_result[epoch, 0, 0], pl_result[epoch, 1, 0])
    output_write(out_str)
    if math.isnan(pl_result[epoch, 0, 0]) or math.isnan(pl_result[epoch, 1, 0]):
        pl_result[epoch:, 0, 0] = float('nan')
        pl_result[epoch:, 1, 0] = float('nan')
        break
np.save(result_path + 'npy/' + result_file_name, pl_result)

# ===============================================================
# === Plotting prepare
# ===============================================================
model.eval()

x_star = torch.from_numpy(X_star[:, 0:1]).type(torch.FloatTensor).cuda()
t_star = torch.from_numpy(X_star[:, 1:2]).type(torch.FloatTensor).cuda()

u_pred = model(net_in(x_star, t_star))
u_pred = u_pred.detach().cpu().numpy()

error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print('Error u: %e' % (error_u))

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
Error = np.abs(Exact - U_pred)

# ===============================================================
# === Plotting
# ===============================================================

fig, ax = newfig(1.0, 1.1)
ax.axis('off')

####### Row 0: u(t,x) ##################
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (u_train.shape[0]), markersize=4,
        clip_on=False)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.legend(frameon=False, loc='best')
ax.set_title('$u(t,x)$', fontsize=10)

####### Row 1: u(t,x) slices ##################
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[25, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = 0.25$', fontsize=10)
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_title('$t = 0.50$', fontsize=10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[75, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_title('$t = 0.75$', fontsize=10)

savefig(result_path + 'plot/' + result_file_name)


