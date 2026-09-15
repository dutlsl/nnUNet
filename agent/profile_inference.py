"""
ViViM+Mamba3 T=3 inference profiling on 4090.
Measures latency, GPU memory, and throughput.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Compatibility patch
import torch
if not hasattr(torch, 'float8_e8m0fnu'):
    torch.float8_e8m0fnu = None
if not hasattr(torch, 'float4_e2m1fn_x2'):
    torch.float4_e2m1fn_x2 = None

import time
import numpy as np
from models.vivim_backbone import VivimBackbone

def profile(T=3, B=1, H=448, W=448, num_warmup=10, num_runs=100):
    device = torch.device('cuda')
    model = VivimBackbone(
        in_channels=1, num_classes=4, base_channels=32,
        d_state=16, d_conv=4, expand=2,
        use_mamba=True, mamba_version='mamba3', mamba_headdim=64,
    ).to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    x = torch.randn(B, T, 1, H, W, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # Memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    # Latency
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    lat = np.array(latencies)
    print(f"\n=== ViViM+Mamba3 Inference Profile (B={B}, T={T}, {H}x{W}) ===")
    print(f"Peak GPU Memory: {peak_mem:.1f} MB")
    print(f"Latency  mean: {lat.mean():.2f} ms | std: {lat.std():.2f} ms")
    print(f"Latency  p50:  {np.percentile(lat, 50):.2f} ms")
    print(f"Latency  p95:  {np.percentile(lat, 95):.2f} ms")
    print(f"Latency  p99:  {np.percentile(lat, 99):.2f} ms")
    print(f"Throughput: {1000.0 / lat.mean():.1f} FPS")
    print(f"30fps budget: {33.3:.1f} ms → headroom: {33.3 - lat.mean():.1f} ms")

if __name__ == "__main__":
    profile(T=3, B=1)
    print()
    profile(T=5, B=1)
