import torch
from torch import Tensor

class MinMaxScaler(object):

    def __init__(
        self,
        dim: int | None,
        *args, **kwargs):
        self.dim = dim
        self.min_index = None
        self.max_index = None

    def fit(self, data: Tensor) -> None:
        if self.dim is None:
            self.min = data.min()
            self.max = data.max()
        else:
            self.min, self.min_index = data.min(dim=self.dim, keepdim=True)
            self.max, self.max_index = data.max(dim=self.dim, keepdim=True)
        return None

    def transform(self, data: Tensor) -> Tensor:
        return (data - self.min) / (self.max - self.min)
    
    def fit_transform(self, data: Tensor) -> Tensor:
        self.fit(data)
        return self.transform(data)

    def __call__(self, data: Tensor) -> Tensor:
        return self.fit_transform(data)
