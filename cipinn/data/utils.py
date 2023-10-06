import scipy.io
import os
import numpy as np
import torch


def load_burgers_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'burgers_shock.mat')
    data = scipy.io.loadmat(path)
    return data


class FixedBounds(object):
    def __init__(self, confidence, u_str, device, folder=None, isChebyshev=False, **kwargs):
        """
        Input:
            confidence: example 0.95 for 95% CI
            u_str: string to get u values from npz files. Example 'u_i" or "u_b".
            path: results of Monte Carlo (MC) simulation
            IsChebyshev: compute empirical Chebyshev's bound from MC results
            kwargs: allow fixed lb and ub values, if path is None

        Outputs:
            lb: lower bound
            ub: upper bound
        """
        self.confidence = confidence
        self.u_str = u_str
        self.folder = folder
        self.isChebyshev = isChebyshev
        self.lb = None
        self.ub = None
        self.mean = None

        # NOTE: lb and ub are arrays
        if folder is None:
            if "lb" in kwargs.keys():
                self.lb = kwargs["lb"]

            if "ub" in kwargs.keys():
                self.ub = kwargs["ub"]

            if "mean" in kwargs.keys():
                self.mean = kwargs["mean"]
        else:
            self.lb, self.ub, self.mean = self._compute_bounds()
            self.lb = self.lb.to(device)
            self.ub = self.ub.to(device)
            self.mean = self.mean.to(device)

    def _get_Chebyshev(self, target, x):
        """
            Getting Chebyshev's inequality from experimental/sample data.
            See eq 2 in Stellato, B., Van Parys, B. P., & Goulart, P. J. (2017).
            Multivariate Chebyshev inequality with estimated mean and variance. The American Statistician, 71(2), 123-127.
        """
        x_stdev = np.std(x)
        N = len(x)

        # consider without the floor function first
        assert target * N**2 > N
        lamb1 = np.sqrt((N**2 - 1) / (target * N**2 - N))

        # verify the eq2 <= target
        eq2rhs = np.floor((N + 1) * (N ** 2 - 1 + N * lamb1 ** 2) / (N ** 2 * lamb1 ** 2)) / (N + 1)
        assert eq2rhs <= target

        return -lamb1 * x_stdev, lamb1 * x_stdev

    def _compute_bounds(self):
        # Get data from MC experiment.
        # Extract u values from every npz file.
        npz_files = os.listdir(self.folder)

        # initialize a data numpy array using 1.npz to record all u values.
        # dim 0 denotes individual u, dim 1 corresponds to values from MC experiment.
        filepath = os.path.join(self.folder, '1.npz')
        data = np.load(filepath)[self.u_str]

        # remove 1.npz from the list and get the records of the remaining npz files.
        npz_files.remove('1.npz')
        for fn in npz_files:
            if fn.endswith(".npz"):
                filepath = os.path.join(self.folder, fn)
                data = np.append(data, np.load(filepath)[self.u_str], axis=1)

        # compute 95% CI from MC data
        mc_high = []
        mc_low = []
        mc_mean = []
        target = 1 - self.confidence
        lower_interval = 0.5 - self.confidence / 2
        upper_interval = 0.5 + self.confidence / 2

        for i in range(data.shape[0]):
            if self.isChebyshev:
                # using Chebyshev's inequality
                lb, ub = self._get_Chebyshev(target, data[i, :])
            else:
                sortedvals = np.sort(data[i, :])
                lb = sortedvals[int(lower_interval * len(sortedvals))]
                ub = sortedvals[int(upper_interval * len(sortedvals))]

            mc_low.append(lb)
            mc_high.append(ub)
            mc_mean.append(np.mean(data[i, :]))

        # reshape to (:, 1)
        mc_high = torch.from_numpy(np.array(mc_high)[:, None]).float()
        mc_low = torch.from_numpy(np.array(mc_low)[:, None]).float()
        mc_mean = torch.from_numpy(np.array(mc_mean)[:, None]).float()

        return mc_low, mc_high, mc_mean
