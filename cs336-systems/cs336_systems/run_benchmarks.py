import transformer_benchmarking_without_pytorch_profiler as bt
from argparse import Namespace
from pathlib import Path
import contextlib
import sys
from common import KOA_SCRATCH_PATH, CS336_SYSTEMS_PATH 

sizes = ["small", "medium", "large", "xl", "2.7B"]

# Run one benchmark configuration
def run_one(size, backward, warmup=1):
    cfg = Namespace(
        size=size,
        ctx_len=128,
        vocab=10_000,
        batch=16,
        steps=5,
        warmup=warmup,
        no_backward=not backward,
    )
    bt.main(cfg)

# output paths
result_dir = CS336_SYSTEMS_PATH / "result"
result_dir.mkdir(parents=True, exist_ok=True)
with_warmup_path = result_dir / "benchmark_with_warmup.txt"
no_warmup_path = result_dir / "benchmark_no_warmup.txt"

# Benchmark WITH warm-up
with with_warmup_path.open("w") as f, contextlib.redirect_stdout(f):
    for s in sizes:
        run_one(s, backward=False, warmup=1)  # forward only
    for s in sizes:
        run_one(s, backward=True, warmup=1)   # forward + backward

# Benchmark WITHOUT warm-up
with no_warmup_path.open("w") as f, contextlib.redirect_stdout(f):
    for s in sizes:
        run_one(s, backward=False, warmup=0)  # forward only
    for s in sizes:
        run_one(s, backward=True, warmup=0)   # forward + backward


print(f"Benchmark complete — results saved to:\n  {with_warmup_path}\n  {no_warmup_path}", file=sys.stderr)
