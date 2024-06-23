import torch
import gc


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()
