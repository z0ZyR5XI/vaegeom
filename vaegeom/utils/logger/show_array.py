import numpy as np
import torch
from torch import Tensor

from vaegeom.utils.io.functions import head_array


class ShowArray(object):
    """Display Array (np.ndarray or torch.Tensor) information to Logger."""

    def __init__(self, logger: 'Logger'):
        self.logger = logger

    def __call__(
        self,
        array: np.ndarray | Tensor,
        name: str | None = None,
        verbose: bool = False) -> None:
        self.head(array, name)
        if verbose:
            self.logger.info(array)
        else:
            self.logger.debug(array)
    
    def head(
        self,
        array: np.ndarray | Tensor,
        name: str | None = None) -> None:
        self.logger.info(head_array(array, name))
