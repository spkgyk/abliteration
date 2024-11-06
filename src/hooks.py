from typing import Callable, Dict, List
from functools import wraps

import torch


def tensor_device_decorator(func):
    """
    A decorator that ensures all PyTorch tensors in the function arguments are on the same device.

    This decorator moves all tensor arguments to the device of the first tensor argument
    before executing the decorated function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The wrapped function that handles device consistency.
    """

    @wraps(func)
    def wrapper(*args):
        device = args[0].device
        args = [arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args]
        return func(*args)

    return wrapper


def direction_ablation_hook(refusal_direction: torch.Tensor) -> Callable:
    """
    Creates a hook function that ablates (removes) a specific direction from activations.

    This function returns a hook that can be used to modify neural network activations
    by removing the component along a specified direction vector.

    Args:
        refusal_direction (torch.Tensor): The direction vector to ablate from the activations.

    Returns:
        Callable: A hook function that can be registered with PyTorch modules.
    """

    @tensor_device_decorator
    def _ablate(activation: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        Removes the component of the activation along the specified direction.

        Args:
            activation (torch.Tensor): The activation tensor to modify.
            direction (torch.Tensor): The direction vector to ablate.

        Returns:
            torch.Tensor: The modified activation with the specified direction removed.
        """
        proj = torch.matmul(activation, direction.unsqueeze(-1)) * direction
        return activation - proj

    # Return a lambda function that applies the _ablate function
    return lambda module, input, output: _ablate(output, refusal_direction)


@tensor_device_decorator
def get_orthogonalized_matrix(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Orthogonalizes a matrix with respect to a vector.

    This function removes the component of the matrix that is parallel to the given vector,
    effectively making the matrix orthogonal to the vector.

    Args:
        matrix (torch.Tensor): The matrix to be orthogonalized.
        vec (torch.Tensor): The vector to orthogonalize against.

    Returns:
        torch.Tensor: The orthogonalized matrix.

    Notes:
        The function handles both cases where the vector length matches either
        the matrix rows or columns by adjusting the computation accordingly.
    """
    if len(vec) == len(matrix):
        proj = torch.matmul(vec, matrix)
    else:
        proj = torch.matmul(matrix, vec)

    # Compute the outer product of vec and proj
    outer_product = torch.outer(vec, proj)

    # Subtract the projection from the original matrix
    if outer_product.shape == matrix.shape:
        result = matrix - outer_product
    else:
        result = matrix - outer_product.T

    return result


def get_activation_hook(activations: Dict[str, List[torch.Tensor]], name: str, position: int, pre: bool = False) -> Callable:
    """
    Creates a hook function to collect activations from a specific position in a neural network layer.

    This function returns a hook that stores activations in a dictionary for later analysis.

    Args:
        activations (Dict[str, List[torch.Tensor]]): Dictionary to store collected activations.
        name (str): Identifier for the layer being hooked.
        position (int): Position in the sequence to collect activations from.
        pre (bool, optional): If True, collect input activations; if False, collect output activations.
            Defaults to False.

    Returns:
        Callable: A hook function that can be registered with PyTorch modules.
    """

    def hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor = None):
        tensor = input[0] if pre else output
        activations[name].append(tensor[:, position, :].detach().cpu())

    return hook
