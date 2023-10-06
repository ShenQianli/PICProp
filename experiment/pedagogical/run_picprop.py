import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
plt.rcParams.update({'font.size': 16})
colors = [mcolor.TABLEAU_COLORS[k] for k in mcolor.TABLEAU_COLORS.keys()]

parser = argparse.ArgumentParser()
parser.add_argument('--region', type=str, default='chi2')
args = parser.parse_args()

for x_q in np.linspace(-1, 1, 41):
    if not os.path.exists('logs/{}/picprop_{}'.format(args.region, round(x_q, 2))):
        os.system('python picprop.py --region {} --x_q {}'.format(args.region, x_q))

low_bound, high_bound = [], []
for x_q in np.linspace(-1, 1, 41):
    low = np.load('logs/{}/picprop_{}/recs/rec-{}-low.npz'.format(args.region, round(x_q, 2), round(x_q, 2)))['obj_traj'][-1]
    high = np.load('logs/{}/picprop_{}/recs/rec-{}-high.npz'.format(args.region, round(x_q, 2), round(x_q, 2)))['obj_traj'][-1]
    low_bound.append(low)
    high_bound.append(high)

x_t = np.linspace(-1, 1, 101).reshape((-1, 1))
gt = np.sin(np.pi * x_t)
plt.plot(x_t, gt, color='black', label='$\sin(\pi x)$')
x_qs = np.linspace(-1, 1, 41)
plt.plot(x_qs, low_bound, color=colors[2], linestyle='--')
plt.plot(x_qs, high_bound, color=colors[2], linestyle='--', label='PICProp')
z = np.load('data/{}.npz'.format(args.region))['z']
plt.scatter([-1., 1.] * len(z), z.reshape((-1, 1)), marker='x', color='black', label='sample', s=19)
plt.xlabel('x')
plt.ylabel('u')
plt.title('{}'.format(args.region))
plt.legend()
plt.tight_layout()
plt.savefig('plots/picprop_{}.pdf'.format(args.region))

