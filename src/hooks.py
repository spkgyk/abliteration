from functools import wraps
from typing import Callable

import torch


def tensor_device_decorator(func):
    """
    Decorator to ensure tensors are on the same device before operations.
    """

    @wraps(func)
    def wrapper(*args):
        device = args[0].device
        args = [arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args]
        return func(*args)

    return wrapper


def direction_ablation_hook(refusal_direction: torch.Tensor) -> Callable:
    """
    Creates a hook function for direction ablation.

    Args:
        refusal_direction (torch.Tensor): The direction to ablate.

    Returns:
        Callable: A hook function that performs direction ablation.
    """

    @tensor_device_decorator
    def _ablate(activation: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        Ablates the activation in the specified direction.

        Args:
            activation (torch.Tensor): The activation tensor.
            direction (torch.Tensor): The direction to ablate.

        Returns:
            torch.Tensor: The ablated activation.
        """
        proj = torch.matmul(activation, direction.unsqueeze(-1)) * direction
        return activation - proj

    # Return a lambda function that applies the _ablate function
    return lambda module, input, output: _ablate(output, refusal_direction)


@tensor_device_decorator
def get_orthogonalized_matrix(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Orthogonalizes a matrix with respect to a vector.

    Args:
        matrix (torch.Tensor): The input matrix to be orthogonalized.
        vec (torch.Tensor): The vector to orthogonalize against.

    Returns:
        torch.Tensor: The orthogonalized matrix.
    """
    # Compute the projection of vec onto each column of the matrix
    proj = torch.matmul(vec, matrix)

    # Compute the outer product of vec and proj
    outer_product = torch.outer(vec, proj)

    # Subtract the projection from the original matrix
    result = matrix - outer_product

    return result
