import dataclasses

import torch
from torch import nn
from torch import Tensor

###############################################################################
#### Dicts ####################################################################

PARAMETRIZATIONS = {
    'spectral_norm': nn.utils.parametrizations.spectral_norm,
    'weight_norm': nn.utils.parametrizations.weight_norm
}

NORMS = {
    'LayerNorm': nn.LayerNorm,
    'BatchNorm1d': nn.BatchNorm1d,
    'BatchNorm': nn.BatchNorm1d
}

class Exp(nn.Module):
    """Activation function: Exponential"""

    def forward(self, input: Tensor) -> Tensor:
        return torch.exp(input)

ACTIVATIONS = {
    'Sigmoid': nn.Sigmoid,
    'Softplus': nn.Softplus,
    'Exp': Exp,
    'GLU': nn.GLU,
    'Tanh': nn.Tanh,
    'ELU': nn.ELU,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU,
    'Swish': nn.SiLU,
    'Mish': nn.Mish,
    'ReLU': nn.ReLU
}

###############################################################################
#### Define class #############################################################

class LinearLayerBlock(nn.Module):
    """
    Single linear layer.
    """

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        bias: bool,
        parametrization: str | None,
        parametrize_kw: dict,
        normalization: str | None,
        norm_kw: dict,
        activation: str | None,
        activ_kw: dict,
        dropout: float,
        *args, **kwargs):
        """
        Parameters (required)
        --------------------- 
        """
        super().__init__()
        self.fc = nn.Linear(dim_input, dim_output, bias=bias)
        if parametrization is not None:
            self.fc = PARAMETRIZATIONS[parametrization](self.fc, **parametrize_kw)
        self.norm = NORMS.get(normalization, nn.Identity)(dim_output, **norm_kw)
        self.activ = ACTIVATIONS.get(activation, nn.Identity)(**activ_kw)
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = self.norm(x)
        x = self.activ(x)
        x = self.dropout(x)
        return x


@dataclasses.dataclass
class LinearLayerConfig:
    dims: list[int] | None = None
    bias: bool | list[bool] = True
    parametrization: str | None | list[str | None] = None
    parametrize_kw: dict | list[dict] = dataclasses.field(default_factory=dict)
    normalization: str | None | list[str | None] = None
    norm_kw: dict | list[dict] = dataclasses.field(default_factory=dict)
    activation: str | None | list[str | None] = None
    activ_kw: dict | list[dict] = dataclasses.field(default_factory=dict)
    dropout: float | list[float] = 0.0

    def __post_init__(self):
        n_layer = 1 if self.dims is None else len(self.dims)
        
        def _create_list(obj):
            if not isinstance(obj, list):
                obj = [obj for _ in range(n_layer)]
            return obj
        
        self.bias = _create_list(self.bias)
        self.parametrization = _create_list(self.parametrization)
        self.parametrize_kw = _create_list(self.parametrize_kw)
        self.norm = _create_list(self.normalization)
        self.norm_kw = _create_list(self.norm_kw)
        self.activation = _create_list(self.activation)
        self.activ_kw = _create_list(self.activ_kw)
        self.dropout = _create_list(self.dropout)


class LinearLayers(nn.Module):
    """
    Multiple linear layers.
    """

    def __init__(
        self,
        dim_input: int,
        config: LinearLayerConfig,
        *args,
        dims: list[int] | None = None,
        **kwargs):
        super().__init__()
        self.dim_input = dim_input
        self.config = config
        if dims is not None:
            self.dims = [dim_input, *dims]
        else:
            self.dims = [dim_input, *config.dims]
        self.dim_output = self.dims[-1]
        self.list_fc = self.create_fc_list(config)

    def forward(self, x: Tensor) -> Tensor:
        for fc in self.list_fc:
            x = fc(x)
        return x

    def create_fc_list(self, config: LinearLayerConfig) -> nn.ModuleList:
        out = nn.ModuleList([
            LinearLayerBlock(*args)
            for args in zip(
                self.dims[:-1],
                self.dims[1:],
                config.bias,
                config.parametrization,
                config.parametrize_kw,
                config.norm,
                config.norm_kw,
                config.activation,
                config.activ_kw,
                config.dropout
            )
        ])
        return out