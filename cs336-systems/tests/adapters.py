#!/usr/bin/env python3
from __future__ import annotations

from typing import Type

from cs336_systems.triton_rmsnorm import RMSNormAutogradFunction, RMSNormTritonFunction
from cs336_systems.rmsnorm_grad import compute_rmsnorm_grad_g, compute_rmsnorm_grad_x
from cs336_systems.ddp_overlap_individual_parameters import DDPIndividualParameters
import torch


def get_rmsnorm_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements RMSNorm.
    The expectation is that this class will implement RMSNorm
    using standard PyTorch operations.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyRMSNormAutogradFunctionClass
    return RMSNormAutogradFunction

def get_rmsnorm_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements RMSNorm
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_rmsnorm_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyTritonRMSNormAutogradFunctionClass
    return RMSNormTritonFunction


def rmsnorm_backward_g_pytorch(
    grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """
    Compute the gradient of the RMSNorm operation pass with respect to g.

    Args:
        grad_output: torch.Tensor
            Gradient of the loss with respect to the output of the RMSNorm operation.
            This has the same shape as x.
        x: torch.Tensor
            Input to the RMSNorm operation. Shape: (*, H)
        g: torch.Tensor
            The g learnable parameter of the RMSNorm layer. Shape: (H,)

    Returns:
        Gradient of the loss with respect to g. Shape: (H,)
    """
    return compute_rmsnorm_grad_g(grad_output, x, g)


def rmsnorm_backward_x_pytorch(
    grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """
    Compute the gradient of the RMSNorm operation pass with respect to x.

    Args:
        grad_output: torch.Tensor
            Gradient of the loss with respect to the output of the RMSNorm operation.
            This has the same shape as x.
        x: torch.Tensor
            Input to the RMSNorm operation. Shape: (*, H)
        g: torch.Tensor
            The g learnable parameter of the RMSNorm layer. Shape: (H,)

    Returns:
        Gradient of the loss with respect to x. Shape: (*, H)
    """
    return compute_rmsnorm_grad_x(grad_output, x, g)


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDPIndividualParameters(module)
    return DDPIndividualParameters(module)



def ddp_individual_parameters_on_after_backward(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    ddp_model.finish_gradient_synchronization()


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    raise NotImplementedError


def ddp_bucketed_on_after_backward(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def ddp_bucketed_on_train_batch_start(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    raise NotImplementedError


def get_sharded_optimizer(
    params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs
) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    raise NotImplementedError
