import os
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import scipy.io

from cipinn.utils.logger import Logger
from cipinn.model.model import FcNet

# folder where individual picprop experiments are stored
rec_root_dir = os.path.join(os.getcwd(), 'logs')

# create dictionary: keys are the folder names and v are indexes
folder_dict = {k: v for v, k in enumerate(os.listdir(rec_root_dir))}

logger = Logger('bo_efficient')

device = torch.device("cuda:0")

xvals = np.round(np.linspace(-1, 1, 41), 2).tolist()
tvals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

x_test_list = []
t_test_list = []
get_xt = True
L = 0

data = []
for x in xvals:
    for t in tvals:
        expr = "_" + str(x) + "_" + str(t) + "_"
        for key in folder_dict.keys():
            if "bo" in key and expr in key:
                rec = np.load(os.path.join(rec_root_dir, key, "recs", 'rec-{}-{}-{}.npz'.format(x, t, "high")))
                preds_list = rec['pred'][:, 0].tolist()
                if get_xt:
                    # the x_t is common for all recs
                    x_t = rec['x_t']
                    x_test_list = x_t[:, 0].tolist()
                    t_test_list = x_t[:, 1].tolist()
                    L = len(preds_list) # compute the length only once
                    get_xt = False
                data.append(np.transpose([[x]*L, [t]*L, x_test_list, t_test_list, [1.0]*L, preds_list]))

                rec = np.load(os.path.join(rec_root_dir, key, "recs", 'rec-{}-{}-{}.npz'.format(x, t, "low")))
                preds_list = rec['pred'][:, 0].tolist()
                data.append(np.transpose([[x]*L, [t]*L, x_test_list, t_test_list, [-1.0]*L, preds_list]))

data = torch.from_numpy(np.array(data).reshape((-1, 6))).type(torch.float32).to(device)
dataset = TensorDataset(data[:, :5], data[:, 5:])
loader = DataLoader(dataset, batch_size=1024, shuffle=True)

data = scipy.io.loadmat('data/burgers_shock.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
X, T = np.meshgrid(x,t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

x_star_list = X_star[:,0:1].flatten().tolist()
t_star_list = X_star[:,1:2].flatten().tolist()

x_test = np.array([np.transpose([x_star_list, t_star_list, x_star_list, t_star_list, [1.0] * L]),
    np.transpose([x_star_list, t_star_list, x_star_list, t_star_list, [-1.0] * L])]).reshape((-1, 5))

X_test = torch.from_numpy(x_test).type(torch.float32).to(device)

model = FcNet(db=20, depth=8, dx=5, dy=1).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

step = 0
for epoch in range(1000):
    for i, d in enumerate(loader):
        x, y = d
        loss = ((model(x) - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        logger.add_metric('train_loss', loss.item())

    pred = model(X_test).detach().cpu().numpy()
    logger.commit(epoch=epoch, step=step)
    if (epoch + 1) % 100 == 0:
        logger.savez(file='rec-{}.npz'.format(epoch + 1), pred=pred)
        logger.save_model(file='model-{}.pkl'.format(epoch+1), model=model)
