import os.path

import torch
import numpy as np
from cipinn.data.utils import load_burgers_data
from cipinn.pde.base import PDE


class BurgersPDE(PDE):
    def __init__(self, path=None):
        super().__init__(input_dim=2, initial_axis=0, region=[[-1., 1.], [0., 1.]])
        self.nu = 0.01/np.pi
        self.x_i = None
        self.u_i = None
        self.x_b = None
        self.u_b = None
        self.x = None
        self.u = None
        self.load_data(path)

    def load_data(self, path):
        data = load_burgers_data(path)
        x = np.array(data['x'])
        t = np.array(data['t'])
        u = np.array(data['usol'])
        self.x_i = np.concatenate([x, np.zeros_like(x)], axis=1)
        self.u_i = np.array(u[:, :1])
        x_b = np.concatenate([-np.ones_like(t), np.ones_like(t)])
        self.x_b = np.concatenate([x_b, np.concatenate([t, t])], axis=1)
        self.u_b = np.concatenate([u[0][:, None], u[-1][:, None]])
        xx, tt = np.meshgrid(x, t)
        self.x = np.concatenate([xx.reshape((-1, ))[:, None], tt.reshape((-1, ))[:, None]], axis=1)
        self.u = u.T.reshape((-1, ))[:, None]

    def f(self, x, model):
        x, t = x[:, 0][:, None].requires_grad_(True), x[:, 1][:, None].requires_grad_(True)
        u = model(torch.cat((x, t), dim=1))
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        return u_t + u * u_x - self.nu * u_xx


if __name__ == '__main__':
    pde = BurgersPDE()
    exact = pde.u.reshape((100, 256))
    import matplotlib.pyplot as plt
    plt.plot(exact[99, :])
    plt.show()
