#!/usr/bin/env python3

import argparse, timeit, math, torch, sys
from torch import nn
from torch.cuda.amp import autocast
from contextlib import nullcontext
from pathlib import Path
from common import KOA_SCRATCH_PATH, CS336_SYSTEMS_PATH, CS336_BASIC_PATH
sys.path.insert(0, str(CS336_BASIC_PATH))

from cs336_basics.model import BasicsTransformerLM

# Model configurations requested by instruction
SIZES = {
    "small":  dict(d_model=768,  d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240,num_layers=32, num_heads=32),
}

# Constants for benchmarking
def run_step(model, batch, backward, mixed):
    ctx = autocast(dtype=torch.float16) if mixed else nullcontext()
    with ctx:
        out = model(batch).sum()
    if backward:
        out.backward()

#    if backward:
def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--size", choices=SIZES, required=True)
        parser.add_argument("--ctx_len", type=int, default=128)
        parser.add_argument("--vocab", type=int, default=10000)
        parser.add_argument("--batch", type=int, default=16)
        parser.add_argument("--steps", type=int, default=5)
        parser.add_argument("--warmup", type=int, default=1)
        parser.add_argument("--no-backward", action="store_true")
        parser.add_argument("--mixed-precision", action="store_true")
        args = parser.parse_args()

    cfg = SIZES[args.size]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(0)
    model = BasicsTransformerLM(
        vocab_size=args.vocab,
        context_length=args.ctx_len,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        attn_pdrop=None,
        residual_pdrop=None
    ).to(device)
    batch = torch.randint(0, args.vocab, (args.batch, args.ctx_len), device=device)
    backward = not args.no_backward

    # Mixed precision flag
    mixed = args.mixed_precision
    if mixed:
        scaler = torch.cuda.amp.GradScaler() if backward else None
    else:
        scaler = None

    # Warm-up
    for _ in range(args.warmup):
        run_step(model, batch, backward, mixed)
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(args.steps):
        t0 = timeit.default_timer()
        run_step(model, batch, backward, mixed)
        torch.cuda.synchronize()
        times.append(timeit.default_timer() - t0)

    t_mean = sum(times) / len(times)
    t_std = (sum((x - t_mean) ** 2 for x in times) / len(times)) ** 0.5
    mode = "forward" if not backward else "fwd+bwd"
    mode += " (MP)" if mixed else ""
    print(f"{args.size:>6} {mode:12s}: {t_mean*1e3:6.2f} ± {t_std*1e3:.2f} ms")

if __name__ == "__main__":
    main()
