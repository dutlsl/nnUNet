"""
Real-time Latency & FPS Benchmark for TemporalUNet Streaming Inference.

Measures:
  1. Preprocessing time (normalization & padding)
  2. Model forward_streaming latency
  3. Postprocessing time (argmax & pupil center extraction)
  4. Total End-to-End Latency (ms) & Achievable FPS
"""
import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from temporal_unet import TemporalUNet


def benchmark_streaming_performance(
    model_folder: str,
    temporal_checkpoint: str,
    num_runs: int = 500,
    warmup_runs: int = 50,
    device: str = 'cuda',
):
    print(f"=== TemporalUNet Real-time Streaming Benchmark ===")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Load Model
    model = TemporalUNet.from_pretrained(
        model_folder=model_folder,
        checkpoint_name='checkpoint_best.pth',
        num_classes=4,
        deep_supervision=False,
        device=torch.device(device),
    )
    
    ckpt = torch.load(temporal_checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    model.to(device)
    model.reset_temporal_state()

    # Dummy raw frame (Simulating 400x640 grayscale camera input)
    raw_frame = np.random.randint(0, 256, (400, 640), dtype=np.uint8)
    mean_val, std_val = 86.45, 39.94

    # Warmup
    print(f"Warming up for {warmup_runs} runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            img_norm = (raw_frame.astype(np.float32) - mean_val) / std_val
            t = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
            t_padded = torch.nn.functional.pad(t, (0, 48, 0, 0), mode='constant', value=0)
            out = model.forward_streaming(t_padded)
            if isinstance(out, (list, tuple)): out = out[0]
            mask = out.argmax(dim=1)[:, :400, :640].squeeze(0).cpu().numpy()

    # Benchmark Loop with CUDA Events for exact GPU timing
    print(f"Running benchmark over {num_runs} frames...")
    
    preprocess_times = []
    model_times = []
    postprocess_times = []
    total_times = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()

            # Preprocessing
            img_norm = (raw_frame.astype(np.float32) - mean_val) / std_val
            t_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
            t_padded = torch.nn.functional.pad(t_tensor, (0, 48, 0, 0), mode='constant', value=0)
            
            t1 = time.perf_counter()

            # GPU Forward Pass
            start_event.record()
            logits = model.forward_streaming(t_padded)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            end_event.record()
            torch.cuda.synchronize()

            t2 = time.perf_counter()

            # Postprocessing (Argmax + Pupil Center)
            pred_mask = logits.argmax(dim=1)[:, :400, :640].squeeze(0).cpu().numpy().astype(np.uint8)
            pupil_pts = np.argwhere(pred_mask == 3)
            centroid = pupil_pts.mean(axis=0) if len(pupil_pts) > 0 else (np.nan, np.nan)

            t3 = time.perf_counter()

            preprocess_times.append((t1 - t0) * 1000.0)
            model_times.append(start_event.elapsed_time(end_event))
            postprocess_times.append((t3 - t2) * 1000.0)
            total_times.append((t3 - t0) * 1000.0)

    # Statistics
    avg_pre = np.mean(preprocess_times)
    avg_model = np.mean(model_times)
    avg_post = np.mean(postprocess_times)
    avg_total = np.mean(total_times)
    fps = 1000.0 / avg_total

    print("\n================ BENCHMARK RESULTS ================")
    print(f"  Preprocessing Latency:  {avg_pre:.2f} ms")
    print(f"  Model Forward Latency: {avg_model:.2f} ms")
    print(f"  Postprocessing Latency: {avg_post:.2f} ms")
    print(f"---------------------------------------------------")
    print(f"  ★ Total End-to-End Latency: {avg_total:.2f} ms / frame")
    print(f"  ★ Maximum Real-time Throughput: {fps:.1f} FPS")
    print("===================================================\n")

    if fps >= 120:
        print("✅ EXCELLENT: Suitable for ultra-high-speed 120Hz/200Hz smart glasses streaming!")
    elif fps >= 60:
        print("✅ GOOD: Suitable for standard 60Hz real-time eye tracking!")
    else:
        print("⚠️ WARNING: TensorRT / FP16 optimization recommended for higher FPS.")


if __name__ == '__main__':
    benchmark_streaming_performance(
        model_folder='/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d',
        temporal_checkpoint='/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/TemporalUNet_v1/checkpoint_best.pth',
        num_runs=500,
        warmup_runs=50,
        device='cuda',
    )
