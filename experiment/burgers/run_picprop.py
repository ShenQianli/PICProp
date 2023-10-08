import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--region', type=str, default='fixed')
parser.add_argument('--x_q', type=float, default=0.2)
parser.add_argument('--t_q', type=float, default=0.0)
args = parser.parse_args()

xvals = np.round(np.linspace(-1, 1, 41), 2).tolist()
tvals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for x_q in xvals:
    for t_q in tvals:
        if not os.path.exists('logs/bo_{}_{}'.format(x_q, t_q)):
            os.system('python picprop.py --region {} --x_q {} --t_q {}'.format(args.region, x_q, t_q))

