# ===== Pure PyTorch RMSNorm =====
import torch
from cs336_systems.rmsnorm_grad import compute_rmsnorm_grad_x, compute_rmsnorm_grad_g
import triton
import triton.language as tl

class RMSNormAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-8):
        # Save inputs for backward
        ctx.save_for_backward(x, weight)
        ctx.eps = eps

        # Normalize x
        norm_squared = torch.mean(x ** 2, dim=-1, keepdim=True)
        norm = torch.sqrt(norm_squared + eps)
        x_norm = x / norm

        # Scale by weight
        out = x_norm * weight.unsqueeze(0)  
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps

        # Compute gradients
        grad_x = compute_rmsnorm_grad_x(grad_output, x, weight, eps)
        grad_g = compute_rmsnorm_grad_g(grad_output, x, weight, eps)

        return grad_x, grad_g, None  
    
# ===== Triton RMSNorm =====
@triton.jit
def rmsnorm_forward_kernel(x_ptr, weight_ptr, output_ptr, eps, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load x and weight
    x_row = tl.load(x_ptr + row_idx * H + offs, mask=mask, other=0.0)
    weight_row = tl.load(weight_ptr + offs, mask=mask, other=0.0)

    # Normalize x
    rms = tl.sqrt(tl.sum(x_row * x_row) / H + eps)
    x_norm = x_row / rms

    #Apply weight
    out = x_norm * weight_row
    tl.store(output_ptr + row_idx * H + offs, out, mask=mask)

@triton.jit
def rmsnorm_backward_kernel(
    grad_out_ptr, x_ptr, weight_ptr, grad_x_ptr, partial_grad_g_ptr,
    eps, H: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load inputs and grad_output
    x = tl.load(x_ptr + row_idx * H + offs, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    grad_out = tl.load(grad_out_ptr + row_idx * H + offs, mask=mask, other=0.0)

    # Compute normalization
    rms = tl.sqrt(tl.sum(x * x) / H + eps)
    rms3 = rms * rms * rms  
    x_normalized = x / rms

    # Compute grad_x
    grad_out_weight = grad_out * weight
    dot = tl.sum(grad_out_weight * x)
    grad_x = (grad_out_weight / rms) - (x * dot / (H * rms3))
    tl.store(grad_x_ptr + row_idx * H + offs, grad_x, mask=mask)

    # Compute partial grad_g
    partial_grad_g = grad_out * x_normalized
    tl.store(partial_grad_g_ptr + row_idx * H + offs, partial_grad_g, mask=mask)


class RMSNormTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-8):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps

        output = torch.empty_like(x)
        H = x.shape[-1]
        x_flat = x.view(-1, H)
        output_flat = output.view(-1, H)
        num_rows = x_flat.shape[0]

        # Launch Triton kernel
        rmsnorm_forward_kernel[(num_rows,)](
            x_flat, weight, output_flat, eps, H=H, BLOCK_SIZE=triton.next_power_of_2(H)
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps

        H = x.shape[-1]
        x_flat = x.view(-1, H)
        grad_output_flat = grad_output.view(-1, H)
        num_rows = x_flat.shape[0]

        # Allocate outputs for gradients
        grad_x = torch.empty_like(x_flat)
        partial_grad_g = torch.empty_like(x_flat)

        # Launch backward kernel
        rmsnorm_backward_kernel[(num_rows,)](
            grad_output_flat, x_flat, weight,
            grad_x, partial_grad_g,
            eps,
            H=H,
            BLOCK_SIZE=triton.next_power_of_2(H)
        )

        # Reshape and reduce grad_g
        grad_x = grad_x.view_as(x)
        partial_grad_g = partial_grad_g.view_as(x)

        grad_g = partial_grad_g.sum(dim=0)

        return grad_x, grad_g, None  