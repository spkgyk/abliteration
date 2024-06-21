import torch

torch.set_grad_enabled(False)
from transformer_lens import HookedTransformer

model = "meta-llama/Meta-Llama-3-8B-Instruct"

torch.cuda.empty_cache()
HookedTransformer.from_pretrained_no_processing(
    model,
    device="cuda",
    dtype=torch.bfloat16,
    default_padding_side="left",
)

print("Got here")
