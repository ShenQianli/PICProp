import numpy as np
import torch
import warnings


class DataGenerator(object):
    def __init__(self, pde, device):
        self.pde = pde
        self.device = device

    def gen_data_i(
            self,
            n,
            random_loc=True,
            sigma=0.05,
    ):
        if hasattr(self.pde, 'x_i') and hasattr(self.pde, 'u_i'):
            if n <= len(self.pde.x_i):
                if random_loc:
                    idx = np.random.choice(np.arange(len(self.pde.x_i)), n, replace=False)
                else:
                    idx = np.linspace(0, len(self.pde.x_i) - 1, n).astype(np.int)
                x_i, u_i = self.pde.x_i[idx], self.pde.u_i[idx]
            else:
                warnings.warn('data_i: Only {} data points available, while {} are required.'.format(len(self.pde.x_i), n))
                x_i, u_i = self.pde.x_i, self.pde.u_i
            x_i, u_i = torch.from_numpy(x_i), torch.from_numpy(u_i)
        else:
            assert self.pde.initial_axis in [0, 1]
            assert self.pde.input_dim == 2
            if random_loc:
                x_i = np.random.random(n)
            else:
                x_i = np.linspace(0, 1, n)
            if self.pde.initial_axis == 0:
                x_i = np.concatenate((np.zeros(n)[:, None], x_i[:, None]), axis=1)
            else:
                x_i = np.concatenate(x_i[:, None], (np.zeros(n)[:, None]), axis=1)
            x_i = self.to_region(x_i)
            x_i = torch.from_numpy(x_i)
            u_i = self.pde.initial_term(x_i)

        noise = torch.randn(u_i.shape) * sigma
        u_i.data = u_i.data + noise

        return x_i.float().to(self.device), u_i.float().to(self.device)

    def gen_data_b(
            self,
            n,
            random_loc=True,
            sigma=0.05,
    ):
        if hasattr(self.pde, 'x_b') and hasattr(self.pde, 'u_b'):
            if n <= len(self.pde.x_b):
                if random_loc:
                    idx = np.random.choice(np.arange(len(self.pde.x_b)), n, replace=False)
                else:
                    idx = np.linspace(0, len(self.pde.x_b) - 1, n).astype(np.int)
                x_b, u_b = self.pde.x_b[idx], self.pde.u_b[idx]
            else:
                warnings.warn('data_b: Only {} data points available, while {} are required.'.format(len(self.pde.x_i), n))
                x_b, u_b = self.pde.x_b, self.pde.u_b
            x_b, u_b = torch.from_numpy(x_b), torch.from_numpy(u_b)
        else:
            if self.pde.input_dim == 1:
                x_b = np.array([[0.], [1.]])
            elif self.pde.input_dim == 2:
                if self.pde.initial_axis == -1:
                    if random_loc:
                        x_b = []
                        for _ in range(n):
                            i = np.random.random()
                            x_b.append([[i, 0.], [i, 1.], [0., i], [1., i]][np.random.randint(4)])
                    else:
                        x_b = []
                        for i in np.linspace(0., 1., n // 4 + 1)[1: -1]:
                            x_b += [[i, 0.], [i, 1.], [0., i], [1., i]]
                        x_b += [[0., 0.], [0., 1.], [1, 0.], [1., 1.]]
                elif self.pde.initial_axis == 0:
                    if random_loc:
                        x_b = []
                        for _ in range(n):
                            i = np.random.random()
                            x_b.append([[i, 0.], [i, 1.]][np.random.randint(2)])
                    else:
                        x_b = []
                        for i in np.linspace(0., 1., n // 2)[1: -1]:
                            x_b += [[i, 0.], [i, 1.]]
                elif self.pde.initial_axis == 1:
                    if random_loc:
                        x_b = []
                        for _ in range(n):
                            i = np.random.random()
                            x_b.append([[0., i], [1., i]][np.random.randint(2)])
                    else:
                        x_b = []
                        for i in np.linspace(0., 1., n // 2)[1: -1]:
                            x_b += [[0., i], [1., i]]
                else:
                    raise NotImplementedError()
                x_b = np.array(x_b)
            else:
                raise NotImplementedError()

            x_b = self.to_region(x_b)
            x_b = torch.from_numpy(x_b)
            u_b = self.pde.boundary_term(x_b)

        noise = torch.randn(u_b.shape) * sigma
        u_b.data = u_b.data + noise

        return x_b.float().to(self.device), u_b.float().to(self.device)

    def gen_data_u(
            self,
            n,
            random_loc=True,
            sigma=0.05,
    ):
        if hasattr(self.pde, 'x') and hasattr(self.pde, 'u'):
            if n <= len(self.pde.x):
                if random_loc:
                    idx = np.random.choice(np.arange(len(self.pde.x)), n, replace=False)
                else:
                    idx = np.linspace(0, len(self.pde.x) - 1, n).astype(np.int)
                x, u = self.pde.x[idx], self.pde.u[idx]
            else:
                warnings.warn(
                    'data_u: Only {} data points available, while {} are required.'.format(len(self.pde.x_i), n))
                x, u = self.pde.x_i, self.pde.u_i
            x, u = torch.from_numpy(x), torch.from_numpy(u)
        else:
            if random_loc:
                x = np.random.random((n, self.pde.input_dim))
            else:
                # grid
                if self.pde.input_dim == 1:
                    x = np.linspace(0., 1., n)[:, None]
                elif self.pde.input_dim == 2:
                    sqrt = int(np.sqrt(n))
                    x = np.linspace(0, 1, sqrt)
                    y = np.linspace(0, 1, sqrt)
                    x = np.broadcast_to(x[None, :, None], (sqrt, sqrt, 1))
                    y = np.broadcast_to(y[:, None, None], (sqrt, sqrt, 1))
                    x = np.concatenate((x, y), axis=-1).reshape((-1, 2))
                else:
                    raise NotImplementedError()
            x = self.to_region(x)
            x = torch.from_numpy(x)
            u = self.pde.u(x)

        noise = torch.randn(u.shape) * sigma
        u.data = u.data + noise

        return x.float().to(self.device), u.float().to(self.device)

    def gen_data_f(
            self,
            n,
            random_loc=True,
            sigma=0.05,
    ):
        if random_loc:
            x_f = np.random.random((n, self.pde.input_dim))
        else:
            # grid
            if self.pde.input_dim == 1:
                x_f = np.linspace(0., 1., n)[:, None]
            elif self.pde.input_dim == 2:
                sqrt = int(np.sqrt(n))
                x = np.linspace(0, 1, sqrt)
                y = np.linspace(0, 1, sqrt)
                x = np.broadcast_to(x[None, :, None], (sqrt, sqrt, 1))
                y = np.broadcast_to(y[:, None, None], (sqrt, sqrt, 1))
                x_f = np.concatenate((x, y), axis=-1).reshape((-1, 2))
            else:
                raise NotImplementedError()
        x_f = self.to_region(x_f)
        x_f = torch.from_numpy(x_f)
        f = self.pde.force_term(x_f)
        noise = torch.randn(f.shape) * sigma
        f.data = f.data + noise

        return x_f.float().to(self.device), f.float().to(self.device)

    def gen_data_t(
            self,
            n,
            random_loc=False,
    ):
        x_t, u_t = self.gen_data_u(n, random_loc=random_loc, sigma=0.)
        f_t = self.pde.force_term(x_t)
        return x_t.float().to(self.device), u_t.float().to(self.device), f_t.float().to(self.device)

    def to_region(self, x):
        l, u = np.array(self.pde.region)[:, 0], np.array(self.pde.region)[:, 1]
        return x * (u - l) + l
