import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import csv
from typing import Tuple

def setup(rank: int, world_size: int, backend: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)

def create_tensor(size_mb: int, device: str) -> torch.Tensor:

    num_elements = int((size_mb * 1024 * 1024) // 4)  # 4 bytes per float32
    return torch.rand((num_elements,), dtype=torch.float32, device=device)

def benchmark_all_reduce(
    rank: int,
    world_size: int,
    backend: str,
    size_mb: int,
    num_warmup: int = 5,
    num_trials: int = 10
) -> float:

    device = "cuda" if backend == "nccl" else "cpu"
    tensor = create_tensor(size_mb, device)
    
    # Warmup 
    for _ in range(num_warmup):
        dist.all_reduce(tensor, async_op=False)
        if device == "cuda":
            torch.cuda.synchronize()
    
    # Benchmark 
    total_time = 0.0
    for _ in range(num_trials):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        dist.all_reduce(tensor, async_op=False)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        total_time += end - start
    
    return total_time * 1000 / num_trials  # Convert to milliseconds

def worker(
    rank: int,
    world_size: int,
    backend: str,
    size_mb: int,
    result_queue: mp.Queue
):
    setup(rank, world_size, backend)
    try:
        avg_time = benchmark_all_reduce(rank, world_size, backend, size_mb)
        result_queue.put((rank, avg_time))
    except Exception as e:
        result_queue.put((rank, float('nan')))

def run_benchmark(
    world_size: int,
    backend: str,
    size_mb: int
) -> float:

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    
    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=worker,
            args=(rank, world_size, backend, size_mb, result_queue)
        )
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    for _ in range(world_size):
        results.append(result_queue.get())
    
    for p in processes:
        p.join()
    
    # Return average across ranks
    return sum(r[1] for r in results) / len(results)

def main():
  
    output_path = "result/distributed_communication_single_node_results.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Benchmark configurations
    backends = [
        ("gloo", "cpu"),
        ("gloo", "cuda"),
        ("nccl", "cuda")
    ]
    sizes_mb = [0.5, 1, 10, 50, 100, 500, 1000]  # 512KB to 1GB
    world_sizes = [2, 4, 6]

    # Check if CSV exists 
    write_header = not os.path.exists(output_path)
    
    # Run benchmarks and save results
    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "backend", "device", "world_size", "size_mb", 
                "mean_time_ms"
            ])
        
        for backend, device in backends:
            if backend == "nccl" and device == "cpu":
                continue  
            
            for world_size in world_sizes:
                for size_mb in sizes_mb:
                    print(f"Running: {backend} {device} {world_size} procs {size_mb}MB")
                    try:
                        avg_time = run_benchmark(world_size, backend, size_mb)
                        writer.writerow([
                            backend, device, world_size, size_mb, avg_time
                        ])
                        f.flush()
                    except Exception as e:
                        print(f"Failed: {e}")
                        continue

if __name__ == "__main__":
    main()
