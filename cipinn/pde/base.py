from abc import ABCMeta, abstractmethod
import torch


class PDE(metaclass=ABCMeta):
    """
    PDE base
    """
    def __init__(
            self,
            input_dim=1,
            initial_axis=-1,
            region=None,
    ):
        """
        :param input_dim:
        :param initial_axis:
                -1: no initial, all boundary
                0: u[0, :] is initial
                1: u[:, 0] is initial
        :param region:
        """
        self.input_dim = input_dim
        self.initial_axis = initial_axis
        self.region = region if region is not None else [[-1., 1.] * input_dim]

    def initial_term(self, x):
        return torch.zeros((x.shape[0], 1))

    def boundary_term(self, x):
        return torch.zeros((x.shape[0], 1))

    def force_term(self, x):
        return torch.zeros((x.shape[0], 1))

    def u(self, x):
        pass

    @abstractmethod
    def f(self, x, model):
        pass
