from typing import Union, Any, Tuple
from jaxtyping import Float

import einops
import torch


# Inference-time intervention hook
def direction_ablation_hook(direction: torch.Tensor):
    def hook(module: torch.nn.Module, input: torch.Tensor, output: Union[torch.Tensor, Tuple, Any]):
        if isinstance(output, tuple):
            output = tuple(_direction_ablation(t, direction) for t in output)
        elif isinstance(output, torch.Tensor):
            output = _direction_ablation(output, direction)
        # If output is neither a tuple nor a Tensor (e.g., DynamicCache), return it unchanged
        return output

    return hook


def _direction_ablation(activation: Union[torch.Tensor, Any], direction: torch.Tensor):
    if not isinstance(activation, torch.Tensor):
        return activation  # Return unchanged if not a tensor

    if activation.device != direction.device:
        direction = direction.to(activation.device)

    proj = einops.einsum(activation, direction.view(-1, 1), "... d_act, d_act single -> ... single") * direction
    return activation - proj


def get_orthogonalized_matrix(
    matrix: Float[torch.Tensor, "... d_model"], vec: Float[torch.Tensor, "d_model"]
) -> Float[torch.Tensor, "... d_model"]:
    print(matrix.shape, vec.shape)
    proj = einops.einsum(matrix, vec.view(-1, 1), "... d_model, d_model single -> ... single") * vec
    return matrix - proj
