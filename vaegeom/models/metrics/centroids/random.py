import torch
from torch import Tensor

from .functions import kmeans_initial

class RandomCentroids(object):
    """
    Random selection for centroids.
    """

    def __init__(
        self,
        num_centroids: int,
        *args, **kwargs):
        self.num_centroids = num_centroids

    def __call__(self, x: Tensor) -> tuple[list[int], Tensor]:
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
        return kmeans_initial(x, self.num_centroids)
        #probs = torch.ones(self.num_centroids, dtype=z.dtype, device=z.device)
        #return torch.multinomial(probs, self.num_centroids).tolist()