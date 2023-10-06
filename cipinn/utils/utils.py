import torch
import random
import numpy as np
from scipy.stats import norm, chi2, f


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def project(x, mean, sigma, confidence, region, eps=1e-6, **kwargs):
    if region in ['norm', 'gaussian']:
        multiplier = min(norm.ppf((1 + confidence) / 2) * sigma / (torch.sqrt(torch.sum((x.data - mean) ** 2)).item() + eps), 1)
        return multiplier * (x.data - mean) + mean
    elif region == 'chi2':
        multiplier = min(np.sqrt(chi2.ppf(confidence, len(x))) * sigma / (torch.sqrt(torch.sum((x.data - mean) ** 2)).item() + eps), 1)
        return multiplier * (x.data - mean) + mean
    elif region == 't2':
        p = len(x)
        n = kwargs.get('n')
        S = kwargs.get('S')
        assert n is not None and S is not None
        k = f.ppf(confidence, p, n-p) * p * (n - 1) / (n - p) / n
        res = x.data - mean
        # print(np.sqrt(k) / (torch.sqrt(torch.matmul(torch.matmul(res.T, torch.inverse(S)), res)).item() + eps))
        multiplier = min(np.sqrt(k) / (torch.sqrt(torch.matmul(torch.matmul(res.T, torch.inverse(S)), res)).item() + eps), 1)
        return multiplier * (x.data - mean) + mean
    elif region == 'union_gaussian':
        p = len(x)
        c = np.power(confidence, 1/p)
        lb = mean - norm.ppf((1 + c) / 2) * sigma
        ub = mean + norm.ppf((1 + c) / 2) * sigma
        return torch.max(torch.min(x.data, ub), lb)
    elif region == 'cross_product':
        return mean + torch.clamp(x.data - mean,
                                  norm.ppf((1 - confidence) / 2) * sigma,
                                  norm.ppf((1 + confidence) / 2) * sigma)
    elif region == "fixed" and "bounds" in kwargs.keys():
        # fixed confidence intervals, such as 95% CI from samples or Chebyshev's inequality
        lb = kwargs['bounds'].lb
        ub = kwargs['bounds'].ub
        return torch.max(torch.min(x.data, ub), lb)  # clipping x.data with lb and ub.
    else:
        raise NotImplementedError('Unknown region {}'.format(region))
