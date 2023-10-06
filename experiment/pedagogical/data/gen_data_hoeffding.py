import numpy as np

np.random.seed(0)
n = 5
bound = 0.15
um1, u1 = np.random.random(2000) * 2 * bound - bound, np.random.random(2000) * 2 * bound - bound
t = np.sqrt(-np.log(np.sqrt(0.05)/2)) * 2 * bound / n
mean_um1, mean_u1 = np.mean(um1[:n]), np.mean(u1[:n])
np.savez('hoeffding.npz',
         z=np.concatenate([um1[:n, None], u1[:n, None]], 1),
         bound=np.array([[mean_um1 - t, mean_um1 + t], [mean_u1 - t, mean_u1 + t]]))