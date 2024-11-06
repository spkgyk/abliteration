import torch
import gc

from enum import Enum


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()


class LayerOutput(Enum):
    RESIDUAL_PRE = "residual pre"
    RESIDUAL_POST = "residual post"
    MLP_POST = "mlp post"
    ATTN_POST = "attn post"
