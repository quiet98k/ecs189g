
from typing import Callable
import torch
from torch import Tensor

def integrated_gradients(
    func: Callable[[Tensor], Tensor],
    inputs: Tensor,
    baselines: Tensor,
    targets: Tensor,
    *,
    n_steps: int,
):
    """ Computes the integrated gradients of the model output with respect to the input.
        The computation uses
        - Left Riemann sum approximation of the integral with @n_steps.

    Args:
        func (Callable[[Tensor], Tensor]): The function to compute the integrated gradients for.
        inputs (Tensor[N_batches, *I_features]): The tensor of inputs.
        baselines (Tensor[N_batches, *I_features]): The tensor of baselines.
        targets (Tensor[N_batches]): The tensor of target classes.
        n_steps (int, optional): The number of steps to use for the approximation.

    Returns:
        Tensor[N_batches, *I_features]: The integrated gradients of the model for the arguments.
    """

    """
    TODO: Implement your integrated gradients computation using here, using
        Left Riemann sum approximation of the integral with @n_steps.

        You are only allowed to use torch. `torch.func` should be useful.
        You are not allowed to use other libraries to compute the integrated gradients.
    """
    raise NotImplementedError("Implement your integrated gradients computation here.")
