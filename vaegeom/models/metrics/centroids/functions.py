import numpy as np
import torch
from torch import Tensor


def kmeans_initial(
    x: Tensor
    num_centroids: int) -> tuple[list[int], Tensor]:
    """
    Initialization of k-means++.

    Parameters (required)
    ---------------------
    x: (n_batch, n_dim), point coordinates

    Returns
    -------
    cent_idx: (num_centroids,), centroid indices
    c: (num_centroids, n_dim), centroid coordinates
    """
    cent_idx = []
    rest_idx = list(range(len(x)))
    j = np.random.choice(rest_idx)
    cent_idx.append(rest_idx.pop(j))
    for i in range(1, num_centroids):
        dist = torch.cdist(x[cent_idx], x[rest_idx]).min(dim=0)[0]
        prob = dist.square()
        _j = torch.multinomial(prob, 1).item()
        j = rest_idx.pop(_j)
        cent_idx.append(j)
    
    return cent_idx, x[cent_idx]
