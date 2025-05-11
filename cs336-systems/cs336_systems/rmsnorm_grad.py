import torch


def compute_rmsnorm_grad_x(grad_out: torch.Tensor, x: torch.Tensor, g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Get hidden dimension
    H = x.shape[-1]
    # Compute RMS norm
    norm = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)  
    
    # First term
    first_term = (grad_out * g) / norm  
    
    # Second term
    gx = grad_out * g  
    dot_product = (gx * x).sum(dim=-1, keepdim=True)  
    second_term = x * (dot_product / (H * norm ** 3))  
    
    grad_x = first_term - second_term  
    return grad_x

def compute_rmsnorm_grad_g(grad_out: torch.Tensor, x: torch.Tensor, g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:

    d = x.shape[-1] 
    # Compute RMS norm 
    norm = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)  
    # Normalize input
    x_normalized = x / norm  
     # Gradient w.r.t. g
    grad_g = (grad_out * x_normalized).sum(dim=tuple(range(x.dim() - 1)))  
    return grad_g
