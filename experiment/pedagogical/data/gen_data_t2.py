import numpy as np
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

mean = 0
sigma = 0.05
confidence = 0.95
n_data = 5
z = np.random.randn(2 * n_data).reshape((n_data, 2)) * sigma
np.savez('t2.npz', z=z)
