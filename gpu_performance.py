import torch
import time
from torch import nn


def detect_gpu():
    print("=== GPU Information ===")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  CUDA Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Memory (GB): {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f}")
    else:
        print("No CUDA-compatible GPU detected.")
    print("=======================")


def benchmark_matmul(dtype):
    print(f"\nMatMul Test - Data Type: {dtype}")
    a = torch.randn((8192, 8192), device='cuda', dtype=torch.float32)
    b = torch.randn((8192, 8192), device='cuda', dtype=torch.float32)

    a = a.to(dtype)
    b = b.to(dtype)

    # Warm-up
    for _ in range(10):
        torch.matmul(a, b)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.4f} seconds")


def benchmark_conv(dtype):
    print(f"\nConvolution Test - Data Type: {dtype}")
    model = nn.Conv2d(128, 256, kernel_size=3, padding=1).to('cuda').to(dtype)
    x = torch.randn((32, 128, 128, 128), device='cuda', dtype=torch.float32).to(dtype)

    # Warm-up
    for _ in range(10):
        model(x)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        model(x)
    torch.cuda.synchronize()

    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.4f} seconds")


def run_all_tests():
    detect_gpu()
    # Removed torch.int4x since it's not a valid dtype
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        benchmark_matmul(dtype)
        benchmark_conv(dtype)


if __name__ == "__main__":
    run_all_tests()