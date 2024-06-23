from transformer_lens.hook_points import HookPoint
from jaxtyping import Float

import einops
import torch


# Inference-time intervention hook
def direction_ablation_hook(
    activation: Float[torch.Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[torch.Tensor, "d_act"],
):
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    proj = einops.einsum(activation, direction.view(-1, 1), "... d_act, d_act single -> ... single") * direction
    return activation - proj
