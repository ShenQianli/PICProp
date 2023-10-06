import torch

from cipinn.pde.base import PDE

PI = 3.141592653589793


class AllanCahnPDE(PDE):
    def __init__(self):
        super().__init__(input_dim=2)
        
    def force_term(self, x):
        x1 = x[:, 0][:, None]
        x2 = x[:, 1][:, None]
        return (PI ** 2) * 0.01 * ((torch.cos(PI * x1) ** 2) * (torch.sin(PI * x2) ** 2) + (torch.sin(PI * x1) ** 2) * (
                torch.cos(PI * x2) ** 2)) + (torch.sin(PI * x1) ** 3) * (torch.sin(PI * x2) ** 3) - torch.sin(
            PI * x1) * torch.sin(PI * x2)

    def u(self, x):
        x1 = x[:, 0][:, None]
        x2 = x[:, 1][:, None]
        return torch.sin(PI * x1) * torch.sin(PI * x2)

    def f(self, x, model):
        x1 = x[:, 0][:, None].requires_grad_(True)
        x2 = x[:, 1][:, None].requires_grad_(True)
        u = model(torch.cat((x1, x2), dim=1))
        u_x1 = torch.autograd.grad(u, x1, grad_outputs=torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        u_x2 = torch.autograd.grad(u, x2, grad_outputs=torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        return 0.01 * ((u_x1 ** 2) + (u_x2 ** 2)) + u ** 3 - u
