import numpy as np
import random

seed = 5
random.seed(seed)
np.random.seed(seed)

mean = 0
sigma = 0.05
confidence = 0.95
n_data = 1
z = np.random.randn(2 * n_data).reshape((n_data, 2)) * sigma
np.savez('chi2.npz', z=z)
