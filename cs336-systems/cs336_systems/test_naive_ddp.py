#!/usr/bin/env python
import os, torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp

# Simple Toy model for testing
class ToyModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, 10, bias=False)
        self.ln   = nn.LayerNorm(10)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(10, out_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Generate random batches
def make_batch(batch, in_dim, num_classes, device, input_dtype=torch.float32, label_dtype=torch.long):
    x = torch.randint(0, num_classes, (batch, in_dim), device=device, dtype=input_dtype) \
        if input_dtype in [torch.long, torch.int] else torch.randn(batch, in_dim, device=device, dtype=input_dtype)
    y = torch.randint(0, num_classes, (batch,), device=device, dtype=label_dtype)
    return x, y

# Run single-process reference
def run_single(steps=5, batch=16, lr=1e-2, in_dim=8, num_classes=4, seed=42, device="cuda"):
    torch.manual_seed(seed)
    model = ToyModel(in_dim, num_classes).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(steps):
        torch.manual_seed(seed + step)
        x, y = make_batch(batch, in_dim, num_classes, device)
        optim.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        optim.step()

    print("\nFinal single-process parameters (first few values):")
    for n, p in model.named_parameters():
        flat = p.detach().cpu().view(-1)
        print(f"{n:20s}  shape={tuple(p.shape)}  {flat[:6].tolist()}...")

    return [p.detach().cpu().clone() for p in model.parameters()]

# Setup distributed environment
def setup_dist(rank, world_size, backend="nccl", device_type="cuda"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    return torch.device(f"{device_type}:{rank}")

# Cleanup distributed environment
def cleanup():
    dist.destroy_process_group()

# DDP worker function
def ddp_worker(rank, world_size,
               steps, batch, lr, in_dim, num_classes, seed,
               backend, device_type, shared):
    device = setup_dist(rank, world_size, backend, device_type)

    torch.manual_seed(seed)
    model = ToyModel(in_dim, num_classes).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # initial weights
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    # Training loop 
    local_bsz = batch // world_size
    for step in range(steps):
        torch.manual_seed(seed + step)
        x_full, y_full = make_batch(
            batch, in_dim, num_classes, device,
            input_dtype=torch.float32, label_dtype=torch.long
        )
        x = x_full[rank * local_bsz : (rank + 1) * local_bsz]
        y = y_full[rank * local_bsz : (rank + 1) * local_bsz]

        optim.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world_size
        optim.step()

    if rank == 0:
        print("\nFinal DDP parameters (first few values of each tensor):")
        for n, p in model.named_parameters():
            flat = p.detach().cpu().view(-1)
            print(f"{n:20s}  shape={tuple(p.shape)}  {flat[:6].tolist()}...")

    final_state = [p.detach().cpu() for p in model.parameters()]
    all_states = [None] * world_size
    dist.all_gather_object(all_states, final_state)
    if rank == 0:
        shared["ddp_params"] = all_states[0]

    cleanup()

# Check if tensors match
def tensors_match(single_list, ddp_list):
    for idx, (s, d) in enumerate(zip(single_list, ddp_list)):
        if not torch.allclose(s, d):  # defaults: atol=1e-8, rtol=1e-5
            print(f"Mismatch in param {idx}")
            return False
    return True

# Main function to run the tests
def main():
    world_size  = 2
    steps       = 5
    batch       = 16
    lr          = 1e-2
    in_dim      = 8
    num_classes = 4
    seed        = 42
    backend, device_type = "nccl", "cuda"

    print("Running single-process reference …")
    single_params = run_single(steps, batch, lr, in_dim, num_classes, seed, device="cuda")

    print(f"\nRunning naïve DDP (world_size={world_size}) …")
    mp.set_start_method("spawn", force=True)
    manager, shared = mp.Manager(), mp.Manager().dict()

    mp.spawn(
        ddp_worker,
        args=(world_size, steps, batch, lr, in_dim, num_classes, seed,
              backend, device_type, shared),
        nprocs=world_size,
        join=True
    )

    ddp_params = shared["ddp_params"]
    print("\nPARAMETERS MATCH!" if tensors_match(single_params, ddp_params)
          else "\nPARAMETERS DIFFER!")

if __name__ == "__main__":
    main()
