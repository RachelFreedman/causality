import numpy as np
import matplotlib.pyplot as plt

m_0 = np.abs(np.load('reward_diffs_120seed0.npy'))
m_1 = np.abs(np.load('reward_diffs_120seed1.npy'))
m_2 = np.abs(np.load('reward_diffs_120seed2.npy'))

l_0 = np.abs(np.load('reward_diffs_324seed0.npy'))
l_1 = np.abs(np.load('reward_diffs_324seed1.npy'))
l_2 = np.abs(np.load('reward_diffs_324seed2.npy'))

m = np.mean(np.array([m_0, m_1, m_2]), axis=0)
l = np.mean(np.array([l_0, l_1, l_2]), axis=0)

plt.hist(m, bins=np.arange(0, 50, step=2), density=True)
plt.xticks(np.arange(0, 50, step=2))
plt.xlabel("Reward differences in pairwise comparisons $r(traj_j) - r(traj_i)$")
plt.ylabel("Frequency")
plt.show()

plt.hist(l, bins=np.arange(0, 50, step=2), density=True)
plt.xticks(np.arange(0, 50, step=2))
plt.xlabel("Reward differences in pairwise comparisons $r(traj_j) - r(traj_i)$")
plt.ylabel("Frequency")
plt.show()

