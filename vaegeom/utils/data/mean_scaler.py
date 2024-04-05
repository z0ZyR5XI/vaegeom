import torch
from torch import Tensor

class MeanScaler(object):

    def __init__(
        self,
        dim: int | None,
        *args, **kwargs):
        self.dim = dim

    def fit(self, data: Tensor) -> None:
        self.mean = data.mean(dim=self.dim, keepdim=True)
        return None

    def transform(self, data: Tensor) -> Tensor:
        return data - self.mean
    
    def fit_transform(self, data: Tensor) -> Tensor:
        self.fit(data)
        return self.transform(data)

    def __call__(self, data: Tensor) -> Tensor:
        return self.fit_transform(data)