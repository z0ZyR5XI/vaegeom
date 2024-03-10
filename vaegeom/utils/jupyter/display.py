import pathlib

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from vaegeom.utils.io.functions import get_memory_str
from vaegeom.utils.io.functions import head_array


def show_head(array: np.ndarray | Tensor, name: str | None = None) -> None:
    print(head_array(array, name))

def show(array: np.ndarray | Tensor, name: str | None = None) -> None:
    show_head(array, name)
    print(array)

def show_dir(dir: pathlib.Path, pattern: str | None = None):
    msg = f'dir: {dir}'
    if pattern is None:
        files = dir.iterdir()
    else:
        files = dir.glob(pattern)
        msg += ' contains {pattern}'
    files = list(files)
    df = pd.DataFrame({
        'type': ['D' if file.is_dir() else 'F' for file in files],
        'name': [file.name for file in files],
        'size': [get_memory_str(file.stat().st_size) for file in files]
    }).sort_values(['type', 'name'])
    df = df.reset_index(drop=True)
    return display(df)
