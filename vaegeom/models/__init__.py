import torch
from torch import nn

#from .decoders import build_decoder
#from .encoders import build_encoder
from .iwae import IWAE
from .vae import VAE

MODELS = {
    'IWAE': IWAE,
    'VAE': VAE
}

def build_model(
    model: str,
    dim_input: int,
    encoder_kw: dict,
    decoder_kw: dict,
    *args, **kwargs) -> nn.Module:
    return MODELS[model](dim_input, encoder_kw, decoder_kw, **kwargs)
    #inst_encoder = build_encoder(dim_input=dim_input, **encoder_kw)
    #inst_decoder = build_decoder(dim_output=dim_input, **decoder_kw)
    #return MODELS[model](inst_encoder, inst_decoder)