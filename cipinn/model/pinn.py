import torch

from cipinn.model.model import FcNet


class PINN(FcNet):
    def __init__(
            self,
            pde,
            hidden_size=32,
            depth=2,
            lamb_i=1.,
            lamb_b=1.,
            lamb_u=1.,
            lamb_f=1.,
    ):
        super().__init__(db=hidden_size, depth=depth, dx=pde.input_dim, dy=1)
        self.pde = pde
        self.input_dim = pde.input_dim
        self.lamb_i = lamb_i
        self.lamb_b = lamb_b
        self.lamb_u = lamb_u
        self.lamb_f = lamb_f
        self.randominit()

    def f(self, x):
        return self.pde.f(x, model=self)

    def loss(self, data_i=None, data_b=None, data_u=None, data_f=None):
        loss = 0.
        for data, lamb in zip([data_i, data_b, data_u], [self.lamb_i, self.lamb_b, self.lamb_u]):
            if data is None:
                continue
            x, y = data
            if not self.training:
                lamb = 1.
            loss = loss + lamb * torch.mean((self.forward(x) - y) ** 2)
        if data_f is not None:
            x, y = data_f
            loss = loss + self.lamb_f * torch.mean((self.f(x) - y) ** 2)

        return loss
