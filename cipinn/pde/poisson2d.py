import torch


from cipinn.pde.base import PDE

PI = 3.141592653589793


class Poisson2D(PDE):
    def __init__(self):
        super().__init__(input_dim=2)

    def u(self, x):
        x1 = x[:, 0][:, None]
        x2 = x[:, 1][:, None]
        return torch.exp(x1) + torch.exp(x2)

    def boundary_term(self, x):
        x1 = x[:, 0][:, None]
        x2 = x[:, 1][:, None]
        return torch.exp(x1) + torch.exp(x2)

    def force_term(self, x):
        x1 = x[:, 0][:, None]
        x2 = x[:, 1][:, None]
        return torch.exp(x1) + torch.exp(x2)

    def f(self, x, model):
        x1 = x[:, 0][:, None].requires_grad_(True)
        x2 = x[:, 1][:, None].requires_grad_(True)
        u = model(torch.cat((x1, x2), dim=1))
        u_x1 = torch.autograd.grad(u, x1, grad_outputs=torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        u_x1x1 = torch.autograd.grad(u_x1, x1, grad_outputs=torch.ones_like(u_x1),
                                     create_graph=True, retain_graph=True)[0]
        u_x2 = torch.autograd.grad(u, x2, grad_outputs=torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        u_x2x2 = torch.autograd.grad(u_x2, x2, grad_outputs=torch.ones_like(u_x2),
                                     create_graph=True, retain_graph=True)[0]

        return u_x1x1 + u_x2x2
