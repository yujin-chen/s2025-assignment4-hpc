#!/usr/bin/env python3
import os, time, torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp
import torch.nn.functional as F
import sys


sys.path.insert(0, os.path.abspath("/home/yujin31/s2025-assignment4-hpc/cs336-basics"))
sys.path.insert(0, os.path.abspath("/home/yujin31/s2025-assignment4-hpc/cs336-systems"))

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from test_naive_ddp import setup_dist, cleanup  

# Model configs for benchmarking
MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    # "xl":     dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    # "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

# Constants for benchmarking
VOCAB_SIZE = 10000
SEQ_LEN    = 128
BATCH_SIZE = 16
STEPS      = 5
SEED       = 42


# Generate random batch
def make_batch(batch, seq_len, num_classes, device, input_dtype=torch.float32, label_dtype=torch.long):
    x = torch.randint(0, num_classes, (batch, seq_len), device=device, dtype=input_dtype) \
        if input_dtype in [torch.long, torch.int] else torch.randn(batch, seq_len, device=device, dtype=input_dtype)
    y = torch.randint(0, num_classes, (batch, seq_len), device=device, dtype=label_dtype)
    return x, y

# Worker function
def benchmark_worker(rank, world_size, model_cfg, shared):
    device = setup_dist(rank, world_size, backend="nccl", device_type="cuda")

    torch.manual_seed(SEED)
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=SEQ_LEN,
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        num_layers=model_cfg["num_layers"],
        norm_type="rms"
    ).to(device)

    optim = AdamW(model.parameters(), lr=1e-3)

    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    total_time, total_comm = 0.0, 0.0
    local_bsz = BATCH_SIZE // world_size

    for step in range(STEPS):
        torch.manual_seed(SEED + step)
        x_full, y_full = make_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, device, input_dtype=torch.long, label_dtype=torch.long)

        x = x_full[rank * local_bsz : (rank + 1) * local_bsz]
        y = y_full[rank * local_bsz : (rank + 1) * local_bsz]

        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.time()

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()

        torch.cuda.synchronize()
        comm_start = time.time()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world_size
        torch.cuda.synchronize()
        comm_time = time.time() - comm_start

        optim.step()
        torch.cuda.synchronize()
        total_time += time.time() - start
        total_comm += comm_time

    if rank == 0:
        shared["step_time"] = total_time / STEPS
        shared["comm_time"] = total_comm / STEPS

    cleanup()

# Run all configurations
def run_all_configs():
    world_size = 2
    mp.set_start_method("spawn", force=True)

    for model_name, cfg in MODEL_CONFIGS.items():
        manager, shared = mp.Manager(), mp.Manager().dict()

        mp.spawn(
            benchmark_worker,
            args=(world_size, cfg, shared),
            nprocs=world_size,
            join=True,
        )

        step_ms = shared["step_time"] * 1000
        comm_ms = shared["comm_time"] * 1000
        overhead = 100 * shared["comm_time"] / shared["step_time"]

        print(f"\nModel: {model_name}")
        print(f"Avg step time: {step_ms:.3f} ms")
        print(f"Avg comm time: {comm_ms:.3f} ms")
        print(f"Comm overhead: {overhead:.2f}%")

if __name__ == "__main__":
    run_all_configs()
