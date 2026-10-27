"""
GAZER architecture stress test: locate the compute bottleneck.

Hypothesis under test (user's): in the Vivim + Mamba3 backbone, the CNN layers
(enc/dec ConvBlocks) dominate compute, not the Mamba3 temporal block.

What is measured (bf16 autocast, eval mode — matching inference; the training
path is measured with grads OFF for timing, the FLOPs profile is the same):

  A. FLOPs profile — hook counter on Conv2d/ConvTranspose2d/Linear + an
     analytical bound for the Mamba3 SSM (d_model=256, d_state=16, expand=2).
  B. Per-module wall time — CNN encoder (448->56), Mamba3 temporal block
     ([B,T,256,56,56]), CNN decoder (56->448), each in isolation.
  C. Full-forward throughput:
     - streaming FIFO path (T=1, decode-last)      -> FPS (120Hz target)
     - T=5 all-frames decode (training path, B=2)  -> windows/s
     - encode-only (bottleneck features, GAZER head input)
     - T sweep on the Mamba3 block (T=1,5,8)       -> linearity check
  D. mamba ON vs OFF (the ablation switch) for the relative SSM cost.

Usage:
    python tools/gazer_bench.py --gpu 0 --config configs/gazer_grpo_t5.yaml
Results: prints a table + writes logs/diagnostics/gazer_bench.json
"""
import os
import sys
import json
import time
import argparse
import statistics

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = None
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = None

from utils.config import load_config
from models.vivim_backbone_multiframe import VivimBackboneMultiFrame

RESULTS = {}


# ---------------------------------------------------------------------------
# FLOPs counting (analytical, per op — the standard dense-matrix count)
# ---------------------------------------------------------------------------
def _hook_conv2d(fn, hook):
    def h(module, inp, out):
        cin = module.in_channels
        cout = module.out_channels
        kh, kw = module.kernel_size
        oc = out.shape
        flops = 2.0 * cin * cout * kh * kw * oc[2] * oc[3] * out.shape[0]
        if module.bias is not None:
            flops += out.numel()
        hook(flops, module)
    return h


def _hook_convtr(fn, hook):
    def h(module, inp, out):
        cin = module.in_channels
        cout = module.out_channels
        kh, kw = module.kernel_size
        oc = out.shape
        flops = 2.0 * cin * cout * kh * kw * oc[2] * oc[3] * out.shape[0]
        if module.bias is not None:
            flops += out.numel()
        hook(flops, module)
    return h


def _hook_linear(fn, hook):
    def h(module, inp, out):
        first = inp[0] if isinstance(inp, (tuple, list)) else inp
        n = first.numel() / first.shape[-1]
        flops = 2.0 * module.in_features * module.out_features * n
        if module.bias is not None:
            flops += out.numel()
        hook(flops, module)
    return h


def attach_flops_counter(model):
    buckets = {}
    state = {}

    def hook(flops, module):
        top = None
        for name, sub in model.named_modules():
            if sub is module:
                top = name.split(".")[0] if name else "unassigned"
                break
        buckets[top] = buckets.get(top, 0.0) + flops
        state.setdefault("total", 0.0)
        state["total"] += flops

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_hook(_hook_conv2d(m, hook), prepend=True)
        elif isinstance(m, torch.nn.ConvTranspose2d):
            m.register_forward_hook(_hook_convtr(m, hook), prepend=True)
        elif isinstance(m, torch.nn.Linear):
            m.register_forward_hook(_hook_linear(m, hook), prepend=True)
    return buckets, state


def mamba3_flops_bound(d_model, d_state, expand, positions):
    """Analytical FLOPs bound for one Mamba3 block at `positions` tokens.

    Dominant terms (dense-matrix count): in_proj (d -> 2*d_inner + dt + C + ...),
    SSM update (2*d_inner*d_state*2 per position), out_proj (d_inner -> d).
    d_inner = d_model*expand. This is a bound, not an exact kernel count —
    the wall-time measurement is the ground truth.
    """
    d_inner = d_model * expand
    per_pos = (2 * d_model * (2 * d_inner + d_model)      # in_proj (B,C,dt,C,dx)
               + 2 * d_inner * d_state * 2                # state update (x2: dt/dx)
               + 2 * d_inner * d_model)                   # out_proj
    return per_pos * positions


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------
def time_fn(fn, iters=30, warmup=10, dtype=torch.bfloat16):
    """Median CUDA-event wall time in ms over `iters` runs (bf16 autocast)."""
    ts = []
    with torch.no_grad():
        for i in range(warmup):
            with torch.amp.autocast("cuda", dtype=dtype):
                fn()
        torch.cuda.synchronize()
        for _ in range(iters):
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ev0.record()
            with torch.amp.autocast("cuda", dtype=dtype):
                fn()
            ev1.record()
            torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.cuda.set_device(args.gpu)
    dev = f"cuda:{args.gpu}"

    m = cfg.model
    H = W = int(cfg.data.target_resolution[0])
    T = int(cfg.data.temporal_window)
    B_stream, B_train = 1, int(cfg.training.batch_size)
    base = m.base_channels
    C_b = base * 8                      # bottleneck channels (256)
    Hb = Wb = H // 8                    # 56

    out_dir = os.path.join(PROJECT_ROOT, "logs", "diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[bench] gpu={args.gpu} T={T} base={base} bottleneck={C_b}@{Hb}x{Wb}")

    # ------------------------------------------------------------------
    # Model (random init is fine for timing; load the real init for realism)
    # ------------------------------------------------------------------
    model = VivimBackboneMultiFrame(
        in_channels=1, num_classes=m.num_classes, base_channels=base,
        d_state=m.mamba.d_state, d_conv=m.mamba.d_conv,
        expand=m.mamba.expand, use_mamba=True,
        mamba_version=cfg.ablation.mamba_version,
        mamba_headdim=m.mamba.headdim, temporal_window=T,
    ).to(dev)
    ckpt = getattr(m, "pretrained_checkpoint", None)
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("network_weights", sd.get("model_state_dict", sd))
        sd = {k[len("backbone."):]: v for k, v in sd.items()
              if k.startswith("backbone.")} or sd
        model.load_state_dict(sd, strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bench] params={n_params / 1e6:.2f}M (mamba3 d_state={m.mamba.d_state}, "
          f"d_conv={m.mamba.d_conv}, expand={m.mamba.expand})")

    x_train = torch.randn(B_train, T, 1, H, W, device=dev)
    x_stream = torch.randn(B_stream, 1, 1, H, W, device=dev)

    # ------------------------------------------------------------------
    # A. FLOPs profile (one T=5 forward, B=1)
    # ------------------------------------------------------------------
    model1 = model  # single-batch FLOPs
    x1 = x_train[:1]
    buckets, state = attach_flops_counter(model1)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model1(x1, output_all_frames=True)
    flops = {k: v for k, v in buckets.items()}
    pos = 1 * T * Hb * Wb              # FLOPs profile is B=1
    flops["temporal_mamba_bound"] = mamba3_flops_bound(
        C_b, m.mamba.d_state, m.mamba.expand, pos)
    cnn_keys = [k for k in flops
                if k in ("enc1", "enc2", "enc3", "bottleneck",
                         "up3", "dec3", "up2", "dec2", "up1", "dec1",
                         "final_cls")]
    cnn_flops = sum(flops[k] for k in cnn_keys)
    RESULTS["flops_T5_B1"] = {k: v / 1e9 for k, v in flops.items()}
    RESULTS["flops_T5_B1"]["_total"] = (cnn_flops + flops["temporal_mamba_bound"]) / 1e9
    RESULTS["flops_T5_B1"]["_cnn_total"] = cnn_flops / 1e9
    RESULTS["flops_T5_B1"]["_mamba_bound"] = flops["temporal_mamba_bound"] / 1e9

    # ------------------------------------------------------------------
    # B. Per-module wall time (B=1, T=5, bf16)
    # ------------------------------------------------------------------
    xf = x1.view(T, 1, H, W)          # [T,1,H,W] batch of T frames
    with torch.no_grad():
        e1 = model.enc1(xf)
        p1 = model.pool1(e1)
        e2 = model.enc2(p1)
        p2 = model.pool2(e2)
        e3 = model.enc3(p2)
        p3 = model.pool3(e3)
        b = model.bottleneck(p3)

    def run_encoder():
        a = model.enc1(xf)
        a = model.pool1(a)
        a = model.enc2(a)
        a = model.pool2(a)
        a = model.enc3(a)
        a = model.pool3(a)
        return model.bottleneck(a)

    t_enc = time_fn(run_encoder)

    b_seq = b.view(1, T, C_b, Hb, Wb)
    def run_mamba():
        return model.temporal_mamba(b_seq)
    t_mamba_T5 = time_fn(run_mamba)

    # T sweep for the mamba block (linearity check; T=8 pads by repeating
    # frames — timing depends only on sequence length)
    t_mamba = {}
    for Ts in (1, 5, 8):
        reps = (Ts + T - 1) // T
        bs = b.repeat(reps, 1, 1, 1)[:Ts].view(1, Ts, C_b, Hb, Wb)
        t_mamba[f"T{Ts}"] = time_fn(lambda s=bs: model.temporal_mamba(s))
    RESULTS["module_time_ms_B1"] = {
        "cnn_encoder_448to56_T5": t_enc,
        "mamba3_T1": t_mamba["T1"], "mamba3_T5": t_mamba["T5"],
        "mamba3_T8": t_mamba["T8"],
    }

    with torch.no_grad():
        bm = model.temporal_mamba(b_seq)          # [1,T,C,56,56]
        e3a = e3.view(1, T, -1, H // 4, W // 4)
        e2a = e2.view(1, T, -1, H // 2, W // 2)
        e1a = e1.view(1, T, -1, H, W)

    def run_decoder():
        outs = []
        for t in range(T):
            d3 = model.up3(bm[:, t])
            d3 = model.dec3(torch.cat([d3, e3a[:, t]], dim=1))
            d2 = model.up2(d3)
            d2 = model.dec2(torch.cat([d2, e2a[:, t]], dim=1))
            d1 = model.up1(d2)
            d1 = model.dec1(torch.cat([d1, e1a[:, t]], dim=1))
            outs.append(model.final_cls(d1))
        return torch.stack(outs, dim=0)

    t_dec = time_fn(run_decoder)
    RESULTS["module_time_ms_B1"]["cnn_decoder_56to448_T5"] = t_dec

    # ------------------------------------------------------------------
    # C. Full-forward throughput
    # ------------------------------------------------------------------
    # C1. T=5 all-frames decode (training path shape), B=2
    t_fwd_T5 = time_fn(lambda: model(x_train, output_all_frames=True), iters=10, warmup=4)
    # C2. encode-only (GAZER policy-head input)
    t_enc_only = time_fn(lambda: model(x_train, encode_only=True), iters=10, warmup=4)
    # C3. streaming FIFO (B=1, T=1, decode-last) — the 120Hz path
    def stream_one():
        model.reset_cache()
        outs = []
        for _ in range(20):
            outs.append(model(x_stream, output_all_frames=False))
        return outs
    t_stream = time_fn(stream_one, iters=5, warmup=2) / 20.0
    # C4. mamba OFF (ablation switch) — same paths
    model.use_mamba = False
    t_fwd_T5_nom = time_fn(lambda: model(x_train, output_all_frames=True), iters=10, warmup=4)
    t_stream_nom = time_fn(stream_one, iters=5, warmup=2) / 20.0
    model.use_mamba = True

    RESULTS["forward_ms"] = {
        "T5_allframes_B2_mamba": t_fwd_T5,
        "T5_allframes_B2_nomamba": t_fwd_T5_nom,
        "encode_only_B2": t_enc_only,
        "streaming_B1_fifo_perframe_mamba": t_stream,
        "streaming_B1_fifo_perframe_nomamba": t_stream_nom,
    }
    RESULTS["throughput"] = {
        "streaming_fps_mamba": 1000.0 / t_stream,
        "streaming_fps_nomamba": 1000.0 / t_stream_nom,
        "train_windows_per_s_B2": 1000.0 / t_fwd_T5 / (T) * T,  # windows/s
    }
    RESULTS["throughput"]["train_windows_per_s_B2"] = 1000.0 / t_fwd_T5

    # effective GFLOP/s per module
    enc_gflops = flops.get("enc1", 0) + flops.get("enc2", 0) + flops.get("enc3", 0) \
        + flops.get("bottleneck", 0)
    dec_gflops = sum(v for k, v in flops.items()
                     if k in ("up3", "dec3", "up2", "dec2", "up1", "dec1", "final_cls"))
    # those counts are per B=1,T=5 forward
    RESULTS["effective_TFLOP_s_B1bf16"] = {
        "cnn_encoder": enc_gflops / 1e12 / (t_enc / 1000.0),
        "cnn_decoder": dec_gflops / 1e12 / (t_dec / 1000.0),
        "mamba3_T5": flops["temporal_mamba_bound"] / 1e12 / (t_mamba_T5 / 1000.0),
    }

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n=== A. FLOPs (B=1, T=5, 448x448) ===")
    for k in sorted(RESULTS["flops_T5_B1"]):
        print(f"  {k:28s} {RESULTS['flops_T5_B1'][k]:10.3f} GFLOP")
    print(f"  CNN: {RESULTS['flops_T5_B1']['_cnn_total']:.1f} GF vs Mamba3 bound "
          f"{RESULTS['flops_T5_B1']['_mamba_bound']:.1f} GF  "
          f"-> CNN:Mamba = "
          f"{RESULTS['flops_T5_B1']['_cnn_total'] / max(1e-9, RESULTS['flops_T5_B1']['_mamba_bound']):.1f}:1")

    print("\n=== B. Module wall time (B=1, T=5, bf16, ms) ===")
    for k, v in RESULTS["module_time_ms_B1"].items():
        print(f"  {k:32s} {v:9.3f} ms")
    tot = t_enc + t_mamba_T5 + t_dec
    print(f"  encoder {t_enc / tot * 100:.0f}% | mamba {t_mamba_T5 / tot * 100:.0f}% | "
          f"decoder {t_dec / tot * 100:.0f}%")

    print("\n=== C. Throughput ===")
    for k, v in RESULTS["forward_ms"].items():
        print(f"  {k:38s} {v:9.3f} ms")
    print(f"  streaming FPS: mamba={1000 / t_stream:.1f}  nomamba={1000 / t_stream_nom:.1f}")

    print("\n=== D. Effective compute (B=1 bf16) ===")
    for k, v in RESULTS["effective_TFLOP_s_B1bf16"].items():
        print(f"  {k:16s} {v:8.3f} TFLOP/s")

    out = os.path.join(out_dir, "gazer_bench.json")
    with open(out, "w") as f:
        json.dump({"config": args.config, "gpu": args.gpu,
                   "params_M": n_params / 1e6, **RESULTS}, f, indent=2)
    print(f"\n[bench] saved {out}")


if __name__ == "__main__":
    main()
