import torch
import numpy as np

from cipinn.pde.base import PDE

PI = np.pi # 3.141592653589793


class PedagogicalPDE(PDE):
    def __init__(self):
        super().__init__()

    def u(self, x):
        return torch.sin(PI * x)

    def f(self, x, model):
        x = x.requires_grad_(True)
        u = model(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True, retain_graph=True)[0]
        return u_xx - u ** 2 * u_x + PI ** 2 * torch.sin(PI * x) + \
               PI * torch.cos(PI * x) * torch.sin(PI * x) ** 2
