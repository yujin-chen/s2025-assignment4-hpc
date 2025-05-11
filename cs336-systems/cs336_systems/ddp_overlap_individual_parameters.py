import torch
import torch.distributed as dist
from typing import List
from torch.profiler import record_function

class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._handles: List = []

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Register gradient hooks
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_hook(param))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):

        with record_function("gradient_synchronization"):
            while self._handles:
                handle = self._handles.pop()
                if isinstance(handle, tuple):  # Gloo
                    with record_function("gloo_wait"):
                        handle[0].wait()
                        handle[1].div_(dist.get_world_size())
                else:  # NCCL
                    with record_function("nccl_wait"):
                        handle.wait()

    def _make_hook(self, param):
        def hook(*_):
            if param.grad is not None:
                if dist.get_backend() == 'gloo':
                    with record_function("gloo_all_reduce"):
                        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
                        self._handles.append((handle, param.grad))
                else:
                    with record_function("nccl_all_reduce"):
                        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
                        self._handles.append(handle)
        return hook
