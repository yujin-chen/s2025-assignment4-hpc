#!/usr/bin/env python3
import os, time, torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp
import torch.nn.functional as F
import sys
from torch.profiler import profile, record_function, ProfilerActivity
from hta.trace_analysis import TraceAnalysis

sys.path.insert(0, os.path.abspath("/home/yujin31/s2025-assignment4-hpc/cs336-basics"))
sys.path.insert(0, os.path.abspath("/home/yujin31/s2025-assignment4-hpc/cs336-systems"))

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from test_naive_ddp import setup_dist, cleanup
from ddp_overlap_individual_parameters import DDPIndividualParameters

# Model configurations
MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
}

# Constants for benchmarking
VOCAB_SIZE = 10000
SEQ_LEN    = 128
BATCH_SIZE = 16
STEPS      = 10
SEED       = 42

# generate random batches
def make_batch(batch, seq_len, num_classes, device, input_dtype=torch.float32, label_dtype=torch.long):
    x = torch.randint(0, num_classes, (batch, seq_len), device=device, dtype=input_dtype) \
        if input_dtype in [torch.long, torch.int] else torch.randn(batch, seq_len, device=device, dtype=input_dtype)
    y = torch.randint(0, num_classes, (batch, seq_len), device=device, dtype=label_dtype)
    return x, y

# Worker function
def benchmark_worker(rank, world_size, model_name, model_cfg, shared, profile_step=None):
    device = setup_dist(rank, world_size, backend="nccl", device_type="cuda")
    
    torch.manual_seed(SEED)
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=SEQ_LEN,
        **model_cfg,
        norm_type="rms"
    ).to(device)
    
    model = DDPIndividualParameters(model)
    optim = AdamW(model.parameters(), lr=1e-3)

    total_time = 0.0
    local_bsz = BATCH_SIZE // world_size

    for step in range(STEPS):
        torch.manual_seed(SEED + step)
        x_full, y_full = make_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, device, 
                                   input_dtype=torch.long, label_dtype=torch.long)
        x = x_full[rank * local_bsz : (rank + 1) * local_bsz]
        y = y_full[rank * local_bsz : (rank + 1) * local_bsz]

        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.time()

        if profile_step == step:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True
            ) as prof:
                with record_function("model_inference"):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
                with record_function("backward"):
                    loss.backward()
                with record_function("grad_sync"):
                    model.finish_gradient_synchronization()
                optim.step()
            
            prof.export_chrome_trace(f"ddp_overlap_trace_{model_name}_rank{rank}.json")
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            model.finish_gradient_synchronization()
            optim.step()

        torch.cuda.synchronize()
        total_time += time.time() - start

    if rank == 0:
        shared["step_time"] = total_time / STEPS

    cleanup()

# Run benchmarking for all model configurations
def run_all_configs():
    world_size = 2
    mp.set_start_method("spawn", force=True)

    for model_name, cfg in MODEL_CONFIGS.items():
        manager, shared = mp.Manager(), mp.Manager().dict()

        # Run benchmark
        mp.spawn(
            benchmark_worker,
            args=(world_size, model_name, cfg, shared, None),
            nprocs=world_size,
            join=True,
        )
        step_ms = shared["step_time"] * 1000
        print(f"\nModel: {model_name}")
        print(f"Avg step time with overlap: {step_ms:.3f} ms")

        # Run profiler on xl model
        if model_name == "xl":
            mp.spawn(
                benchmark_worker,
                args=(world_size, model_name, cfg, shared, STEPS-1), 
                nprocs=world_size,
                join=True,
            )
            
if __name__ == "__main__":
    run_all_configs()
