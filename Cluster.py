import numpy as np
from sklearn.cluster import KMeans
import torch


class ClusterTry:
    def __init__(self, processed_states):
        self.states = processed_states
        self.K = 5

    def cluster(self):
        assert len(self.states) == 2
        mean_obs = np.mean(self.states, axis=0)
        std_obs = np.std(self.states, axis=0)
        max_obs = np.max(self.states, axis=0)
        min_obs = np.min(self.states, axis=0)
        # k_means = KMeans(n_clusters=self.K, max_iter=500, n_jobs=4).fit(self.states)
        # return k_means.cluster_centers_.flatten()
        return np.concatenate([mean_obs, std_obs, max_obs, min_obs], 0)


cur_states = torch.tensor([[0.0000, 0.9500],
        [0.0000, 0.9500],
        [2.0000, 0.7000],

        [0.0000, 0.9500],
        [0.0000, 0.9500],
        [0.0000, 0.9500]])
b = torch.tensor([0.9500, 0.1])
c = [b for _ in range(5)]

# print(list(b.shape))
size_array = np.array([(x * 5) / (y * 2) if y > 0 else 0
                           for i, (x, y) in enumerate(zip(cur_states[:, 0:1].numpy(), cur_states[:, 1:2].numpy()))])
print(cur_states[:, 0:1]*cur_states[:,1:2])