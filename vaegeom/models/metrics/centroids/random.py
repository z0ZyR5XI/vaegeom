import torch
from torch import Tensor

class RandomCentroids(object):
    """
    Random selection for centroids.
    """

    def __init__(
        self,
        num_centroids: int,
        *args, **kwargs):
        self.num_centroids = num_centroids

    def __call__(self, z: Tensor) -> list[int]:
        """
        Parameters (required)
        ---------------------
        z: (n_batch, n_dim), points in latent space

        Returns
        -------
        out: (n_centroids,), indices of centroids
        """
        probs = torch.ones(self.num_centroids, dtype=z.dtype, device=z.device)
        return torch.multinomial(probs, self.num_centroids).tolist()