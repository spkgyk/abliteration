import einops
import torch


# Inference-time intervention hook
def direction_ablation_hook(refusal_direction: torch.Tensor):
    def hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        return _direction_ablation(output, refusal_direction)

    return hook


def _direction_ablation(activation: torch.Tensor, refusal_direction: torch.Tensor):
    if activation.device != refusal_direction.device:
        refusal_direction = refusal_direction.to(activation.device)
    proj = einops.einsum(activation, refusal_direction.view(-1, 1), "... d_act, d_act single -> ... single") * refusal_direction
    return activation - proj


def get_orthogonalized_matrix(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    proj = einops.einsum(matrix, vec, "d_model n, d_model -> n")
    result = matrix - einops.einsum(vec, proj, "d_model, n -> d_model n")
    return result
