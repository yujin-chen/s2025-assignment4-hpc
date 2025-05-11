import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import sys
from pathlib import Path
from common import CS336_BASIC_PATH, CS336_SYSTEMS_PATH
sys.path.insert(0, str(CS336_BASIC_PATH))
from cs336_basics.model import BasicsTransformerLM
from cs336.basics.optimizer import AdamW

# model configurations requested by instruction
SIZES = {
    "xl": dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
}

# Profiling step
def run_step(model, batch, optimizer):
    with record_function("forward_pass"):
        output = model(batch).sum()

    with record_function("backward_pass"):
        output.backward()

    with record_function("optimizer_step"):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

# Main profiling logic
def main(profile_steps=10, warmup_steps=3, batch_size=16):
    cfg = SIZES["xl"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=128,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        attn_pdrop=None,
        residual_pdrop=None
    ).to(device)
    batch = torch.randint(0, 10000, (batch_size, 128), device=device)
    optimizer = AdamW(model.parameters())

    # Warm-up
    for _ in range(warmup_steps):
        run_step(model, batch, optimizer)
        torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True
    ) as prof:
        for _ in range(profile_steps):
            run_step(model, batch, optimizer)
            torch.cuda.synchronize()
            prof.step()


    out_path = CS336_SYSTEMS_PATH / "result/lm_profiler_stacks.txt"

    # Save 
    prof.export_stacks(str(out_path), "self_cuda_time_total")

    # Fix the file manually by adding semicolons
    fixed_out_path = CS336_SYSTEMS_PATH / "result/lm_profiler_stacks_fixed.txt"
    with open(out_path, "r") as fin, open(fixed_out_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            if " " not in line:
                continue
            stack, count = line.rsplit(" ", 1)
            fixed_stack = stack.replace(" ", ";")
            fout.write(f"{fixed_stack} {count}")

    
    # Print both CPU and CUDA sorted tables
    print("Sorted by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    print("\nSorted by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    main(profile_steps=args.profile_steps, 
         warmup_steps=args.warmup_steps,
         batch_size=args.batch_size)
