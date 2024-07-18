from typing import Callable

import torch


def direction_ablation_hook(refusal_direction: torch.Tensor) -> Callable:
    def _ablate(activation: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        direction = direction.to(activation.device)
        proj = activation @ direction.unsqueeze(-1) * direction
        return activation - proj

    return lambda module, input, output: _ablate(output, refusal_direction)


def get_orthogonalized_matrix(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    # Compute the projection of vec onto each column of the matrix
    proj = torch.matmul(vec, matrix)

    # Outer product of vec and proj using torch.outer
    outer_product = torch.outer(vec, proj)

    # Subtract the projection from the original matrix
    result = matrix - outer_product

    return result
