import torch
import torch.nn as nn
import time
import sys
from torch.profiler import profile
import torch
from torch.cuda.amp import autocast


from common import CS336_BASIC_PATH, CS336_SYSTEMS_PATH
import os
sys.path.insert(0, os.path.abspath("/home/yujin31/s2025-assignment4-hpc/cs336-basics"))
sys.path.insert(0, os.path.abspath("/home/yujin31/s2025-assignment4-hpc/cs336-systems"))



from cs336_basics.model import RMSNorm, RMSNormTriton, BasicsTransformerLM

# Model configurations
SIZES = {
    "small":  dict(d_model=768,  d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

# Constants for benchmarking
def benchmark_norms(benchmark_backward=False):
    # Parameters
    batch_size = 50000
    hidden_sizes = [1024, 2048, 4096, 8192]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps = 1000
    warmup_steps = 10

    # Results
    results = []

    for hidden_dim in hidden_sizes:
        # Create random inputs
        x = torch.randn(batch_size, hidden_dim, device=device, requires_grad=benchmark_backward)

        # RMSNorm setup
        rmsnorm = RMSNorm(hidden_dim).to(device)
        layernorm = nn.LayerNorm(hidden_dim).to(device)
        triton_rmsnorm = RMSNormTriton(hidden_dim).to(device)

        # Create fake gradient for backward
        if benchmark_backward:
            dy = torch.randn_like(x)

        # Warmup
        for _ in range(warmup_steps):
            _ = rmsnorm(x)
            _ = layernorm(x)
            _ = triton_rmsnorm(x)
        torch.cuda.synchronize()

        # Benchmark RMSNorm
        start = time.time()
        for _ in range(steps):
            if benchmark_backward and x.grad is not None:
                x.grad = None
            y = rmsnorm(x)
            if benchmark_backward:
                y.backward(dy)
        torch.cuda.synchronize()
        rmsnorm_time = (time.time() - start) / steps * 1000  # ms per step

        # Benchmark LayerNorm
        start = time.time()
        for _ in range(steps):
            if benchmark_backward and x.grad is not None:
                x.grad = None
            y = layernorm(x)
            if benchmark_backward:
                y.backward(dy)
        torch.cuda.synchronize()
        layernorm_time = (time.time() - start) / steps * 1000  # ms per step

        # Benchmark Triton RMSNorm
        start = time.time()
        for _ in range(steps):
            if benchmark_backward and x.grad is not None:
                x.grad = None
            y = triton_rmsnorm(x)
            if benchmark_backward:
                y.backward(dy)
        torch.cuda.synchronize()
        triton_rmsnorm_time = (time.time() - start) / steps * 1000  # ms per step

        # Save
        results.append((hidden_dim, rmsnorm_time, layernorm_time, triton_rmsnorm_time))

    # Print result
    if benchmark_backward:
        print("\n=== RMSNorm vs LayerNorm vs Triton RMSNorm Timing (FORWARD + BACKWARD, ms) ===")
    else:
        print("\n=== RMSNorm vs LayerNorm vs Triton RMSNorm Timing (FORWARD ONLY, ms) ===")

    print(f"{'Hidden dim':>12} | {'RMSNorm (ms)':>12} | {'LayerNorm (ms)':>14} | {'Triton RMS (ms)':>15}")
    for dim, rms_t, layer_t, triton_rms_t in results:
        print(f"{dim:12d} | {rms_t:12.4f} | {layer_t:14.4f} | {triton_rms_t:15.4f}")


# Benchmark different model sizes
def benchmark_different_model_size(norm_type="rms", benchmark_backward=False, compile_model=False, memory_profile=False, model_size="all" , use_bfloat16=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for size_name, cfg in SIZES.items():
        if model_size != "all" and size_name != model_size:
            continue  

        model = BasicsTransformerLM(
            vocab_size=10000,
            context_length=128,
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            norm_type=norm_type,
        ).to(device)

        if compile_model:
            model = torch.compile(model)

        batch = torch.randint(0, 10000, (16, 128), device=device)

        if memory_profile:
            run_memory_profile(
                model,
                batch,
                output_prefix=f"{size_name}_{norm_type}_{'compiled' if compile_model else 'vanilla'}_{'fulltraining' if benchmark_backward else 'forwardonly'}",
                full_training=benchmark_backward,
                use_bfloat16=use_bfloat16,
            )
            continue

        if benchmark_backward:
            output = model(batch)
            dy = torch.randn_like(output)

        # Warmup
        for _ in range(1):
            output = model(batch)
            if benchmark_backward:
                output.backward(dy)
            torch.cuda.synchronize()

        # Timing
        times = []
        for _ in range(5):
            model.zero_grad()

            start = time.time()
            output = model(batch)
            if benchmark_backward:
                output.backward(dy)
            torch.cuda.synchronize()

            end = time.time()
            times.append((end - start) * 1000)  # ms

        mean_time = sum(times) / len(times)
        mode = "forward+backward" if benchmark_backward else "forward"
        compiled_tag = "compiled" if compile_model else "vanilla"
        print(f"{size_name:>6} {mode:>15} ({compiled_tag} {norm_type}): {mean_time:.2f} ms")
        torch.cuda.empty_cache()

# Memory profiling
def run_memory_profile(model, batch, n_steps=3, output_prefix="memory_profile", device="cuda", full_training=False, use_bfloat16=False):
    # Enable memory recording
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    optimizer = None
    if full_training:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=n_steps),
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(n_steps):
            if use_bfloat16:
                # USE bfloat16 autocast context
                with autocast(dtype=torch.bfloat16):
                    output = model(batch)
                    dummy_loss = output.sum()
            else:
                # Normal float32
                output = model(batch)
                dummy_loss = output.sum()

            _ = dummy_loss.item()  # force compute

            if full_training:
                dummy_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            prof.step()





    # Make sure result directory exists
    os.makedirs(str(CS336_SYSTEMS_PATH / "result"), exist_ok=True)

    # Construct full paths
    timeline_path = str(CS336_SYSTEMS_PATH / "result" / f"{output_prefix}_full_train_{full_training}_timeline.html")
    snapshot_path = str(CS336_SYSTEMS_PATH / "result" / f"{output_prefix}_full_train_{full_training}_snapshot.pickle")

    # Save memory snapshot 
    torch.cuda.memory._dump_snapshot(snapshot_path)

 
    try:
        prof.export_memory_timeline(timeline_path, device=device)
        print(f"Memory timeline saved to {timeline_path}")
    except Exception as e:
        print(f"Warning: Could not save memory timeline - {str(e)}")

    # Print peak memory usage
    peak_memory_bytes = torch.cuda.max_memory_allocated(device=device)
    peak_memory_gb = peak_memory_bytes / (1024 ** 3)
    print(f"Peak memory during {output_prefix}: {peak_memory_gb:.2f} GB")

    # Reset peak memory counter
    torch.cuda.reset_peak_memory_stats(device=device)

    print(f"Memory profiling done. Snapshot saved as {snapshot_path}")



if __name__ == "__main__":
    # # First RMSNorm
    # benchmark_different_model_size(norm_type="rms", benchmark_backward=False)
    # print()
    # # Then LayerNorm
    # benchmark_different_model_size(norm_type="layer", benchmark_backward=False)
    # print()
    # # Then Triton RMSNorm
    # benchmark_different_model_size(norm_type="rms_triton", benchmark_backward=False)

    # benchmark_different_model_size(norm_type="rms",benchmark_backward=True)
    # benchmark_different_model_size(norm_type="layer",benchmark_backward=True)
    # benchmark_different_model_size(norm_type="rms_triton", benchmark_backward=True)

    # benchmark_norms(benchmark_backward=False)
    # benchmark_norms(benchmark_backward=True)

    # benchmark_different_model_size(norm_type="rms", benchmark_backward=False, compile_model=False)
    # benchmark_different_model_size(norm_type="rms", benchmark_backward=False, compile_model=True)
    # benchmark_different_model_size(norm_type="rms", benchmark_backward=True, compile_model=False)
    # benchmark_different_model_size(norm_type="rms", benchmark_backward=True, compile_model=True)

    # benchmark_different_model_size(norm_type="rms", benchmark_backward=False, compile_model=False, memory_profile=True, model_size="2.7B")
    # benchmark_different_model_size(norm_type="rms", benchmark_backward=True, compile_model=False, memory_profile=True, model_size="2.7B")
    # benchmark_different_model_size(norm_type="rms", benchmark_backward=False, compile_model=False, memory_profile=True)
    # benchmark_different_model_size(norm_type="rms", benchmark_backward=True, compile_model=False, memory_profile=True)


    benchmark_different_model_size(norm_type="rms", benchmark_backward=False, compile_model=False, memory_profile=True, model_size="2.7B", use_bfloat16=True)
    benchmark_different_model_size(norm_type="rms", benchmark_backward=True, compile_model=False, memory_profile=True, model_size="2.7B", use_bfloat16=True)
