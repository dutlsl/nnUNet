"""
Step 4 — Lightweighting + Annealing experiment (plan §5 step 4).

Purpose (user-chosen objective): maximize lightweighting WHILE the RL reward
is preserved — i.e. keep the per-pixel teacher agreement
(val/agreement_teacher, the SAME metric train_gazer.validate scores) within
-0.01 of the full model, while minimizing params / CNN FLOPs. The inference
target is a single RTX 4090; the full model already streams 198.7 FPS on a
3090, so the real constraint is the RL-reward gate, and the annealing
targets are BOTH CNN sides of the bottleneck (step-2 stress test: CNN is
21.4x the Mamba3 FLOPs; encoder CNN 95.3 GF + decoder 216.1 GF at 448^2 =
95.5% of all CNN FLOPs; Mamba3 is only 13% of wall time and nearly T-flat).

Method — annealed distillation of the RL-tuned policy (gazer_ep9) over a
2x2 (enc_base, dec_base) grid:
  * teacher  = gazer_ep9 backbone (RL-tuned), FROZEN (provides p_t + eye boxes)
  * student  = same backbone with `dec_base` < base_channels (decoder
               channel annealing: each dec stage at dec_base*{4,2,1}) and/or
               `enc_base` < base_channels (encoder channel annealing:
               enc1/2/3 at enc_base*{1,2,4}); a 1x1 skip adapter bridges any
               width gap between encoder skip and decoder input. The
               bottleneck OUTPUT stays base_channels*8 (the Mamba3 temporal
               input) so the learned temporal capacity is inherited; when
               enc_base narrows, the bottleneck INPUT changes (fresh weights,
               trained).
  * grid     = e32d16 -> e32d8 -> e16d16 -> e16d8. The target is chosen by
               the gate's measured degradation tolerance (a gradual fade-out
               coverable by RL+SAM3 tuning), not by static bottleneck
               location.
  * warm-start chain (gradual capacity fade-out): each variant starts from
               the PREVIOUS variant's best checkpoint (shape-aware load);
               the first variant starts from gazer_ep9 itself.
  * loss     = alpha_ce * CE(p_s, SAM3 landscape)  [the RL reward, per-pixel]
               + alpha_kl * KL(p_s || p_t)         [exact per-pixel, to the RL policy]
               + alpha_tv(epoch) * TV(logits_s)    [annealing regularizer]
    (losses/distill_loss.py) — identical per-pixel normalization to GRPO.
  * gate     = val/agreement_teacher within -gate of the teacher's score.

Landscape (RAM-only, no disk supervision data; plan §7.0): Phase 0 precompute
of the SAM3 reward landscape on the TRAIN frames (the student is distilled on
train; val frames are resolved lazily by the live teacher at gate time).
Sequence_PseudoLabels_400 is NEVER used — only the gazer_ep9 checkpoint is.

Usage:
    # static radar (params + decoder FLOPs + decoder wall time + streaming
    # latency per rung) — no landscape, fast:
    python tools/gazer_distill.py --config configs/gazer_grpo_t5.yaml \
        --distill-config configs/gazer_distill_t5.yaml --gpu 0 --static

    # full experiment (landscape precompute + distill per rung + gate):
    python tools/gazer_distill.py --config configs/gazer_grpo_t5.yaml \
        --distill-config configs/gazer_distill_t5.yaml --gpu 0
"""
import os
import re
import sys
import json
import time
import copy
import argparse
import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# PyTorch compat shim (mamba_ssm import chain)
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = None
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = None

from utils.config import load_config
from utils.seed import set_seed
from datasets.sequence_label_dataset import get_sequence_gazer_dataloaders
from models.vivim_gazer import VivimGazer
from rl.grpo import dice_agreement
from losses.distill_loss import DistillLoss
from train_gazer import (build_model, precompute_reward_landscape,
                         GazerTeacherRouter, freeze_bn)
from tools.gazer_bench import attach_flops_counter, time_fn

GAZER_BEST = os.path.join(
    PROJECT_ROOT, "nnUNet_results/gazer_grpo_t5_20261026_103944/checkpoint_best.pth")

DEC_STAGES = ("up3", "dec3", "up2", "dec2", "up1", "dec1", "final_cls")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_backbone_state(model: VivimGazer, ckpt_path: str, shape_aware: bool = False):
    """Load a backbone state dict (flat or backbone.-prefixed) into
    model.backbone; report missing/unexpected.

    shape_aware=True (narrow student): drop keys whose shape differs from
    the target (the narrower decoder + the new skip adapters stay freshly
    initialized) — load_state_dict(strict=False) raises on shape mismatch.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    if not sd:
        sd = ck.get("model_state_dict", sd)
    if shape_aware:
        tgt = model.backbone.state_dict()
        sd = {k: v for k, v in sd.items() if k in tgt and tgt[k].shape == v.shape}
    missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
    print(f"[distill] {os.path.basename(ckpt_path)}: missing={len(missing)} "
          f"unexpected={len(unexpected)}", flush=True)
    return model


def build_teacher(cfg, device: str) -> VivimGazer:
    """RL-tuned policy (gazer_ep9), frozen. Provides p_t + the eye-box source."""
    model = build_model(cfg).to(device)        # semantic init first
    load_backbone_state(model, GAZER_BEST)
    model.eval()
    freeze_bn(model)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_student(cfg, dec_base: int, device: str,
                  enc_base: int = None, init_ckpt: str = None) -> VivimGazer:
    """Narrow student initialized shape-aware from `init_ckpt` (default: the
    RL-tuned policy gazer_ep9; the warm-start chain passes the previous
    rung's best checkpoint). Matching-shape keys load; the new narrow
    stages + skip adapters stay freshly initialized. Annealing starts from
    the RL-tuned policy, not from the semantic fold_1 init.

    enc_base=None keeps the full-width encoder (bottleneck output stays
    base_channels*8 so the inherited Mamba3 weights load)."""
    m = cfg.model
    model = VivimGazer(
        in_channels=1, num_classes=m.num_classes, base_channels=m.base_channels,
        d_state=m.mamba.d_state, d_conv=m.mamba.d_conv, expand=m.mamba.expand,
        mamba_headdim=m.mamba.headdim, use_mamba=cfg.ablation.use_mamba,
        mamba_version=cfg.ablation.mamba_version,
        temporal_window=cfg.data.temporal_window,
        use_gazer=cfg.ablation.use_gazer,
        dec_base=dec_base,
        enc_base=enc_base,
    ).to(device)
    load_backbone_state(model, init_ckpt or GAZER_BEST, shape_aware=True)
    return model


# ---------------------------------------------------------------------------
# Static radar: params + per-variant CNN FLOPs + decoder wall time +
# streaming latency
# ---------------------------------------------------------------------------
def decoder_wall_time_ms(bb: nn.Module, T: int, H: int, W: int,
                         iters: int = 20) -> float:
    """Median wall time (ms) of the DECODER only (bottleneck + skips fed from
    a frozen encode pass), bf16 — mirrors gazer_bench's decoder isolation.
    `bb` is the VivimBackboneMultiFrame (VivimGazer.backbone)."""
    dev = next(bb.parameters()).device
    with torch.no_grad():
        xf = torch.randn(T, 1, H, W, device=dev)
        e1 = bb.enc1(xf); p1 = bb.pool1(e1)
        e2 = bb.enc2(p1); p2 = bb.pool2(e2)
        e3 = bb.enc3(p2); p3 = bb.pool3(e3)
        b = bb.bottleneck(p3)
        b_seq = b.view(1, T, -1, H // 8, W // 8)
        bm = bb.temporal_mamba(b_seq) if bb.use_mamba else b_seq
        e3a = e3.view(1, T, -1, H // 4, W // 4)
        e2a = e2.view(1, T, -1, H // 2, W // 2)
        e1a = e1.view(1, T, -1, H, W)

    def run_decoder():
        outs = []
        for t in range(T):
            outs.append(bb._decode_single_frame(
                bm[:, t], e3a[:, t], e2a[:, t], e1a[:, t]))
        return torch.stack(outs, dim=0)

    return time_fn(run_decoder, iters=iters, warmup=6)


def streaming_latency_ms(bb: nn.Module, H: int, W: int, n: int = 20) -> float:
    """Per-frame streaming FIFO latency (T=1 decode-last), bf16 — the 4090
    real-time path. (Measured on this GPU; the 4090 is faster, so this is an
    upper bound on the demo-environment latency.) `bb` is the
    VivimBackboneMultiFrame (VivimGazer.backbone)."""
    bb.eval()
    dev = next(bb.parameters()).device
    x_stream = torch.randn(1, 1, 1, H, W, device=dev)

    def stream_one():
        bb.reset_cache()
        for _ in range(n):
            bb(x_stream, output_all_frames=False)

    return time_fn(stream_one, iters=5, warmup=2) / n


def variant_grid(dist_cfg, base: int) -> list:
    """[(tag, enc_base, dec_base)] — the 2x2 annealing grid.

    The target is chosen by the RL-reward gate's measured degradation
    tolerance (not by static FLOPs ranking), so the grid must include the
    ENCODER as an annealing candidate too (encoder CNN FLOPs at full
    resolution: enc1 19.1 + enc2 27.7 + enc3 27.7 GF). The bottleneck
    output stays base_channels*8 (the Mamba3 input), so the temporal
    pathway weights are inherited.

    enc_base entries equal to `base` (or None) mean "keep the full-width
    encoder"; the tag is then `e{base}d{dec}`."""
    if dist_cfg.get("grid"):
        grid = [tuple(g) for g in dist_cfg["grid"]]
    else:  # legacy `rungs` (decoder-only) fallback
        grid = [(None, int(r)) for r in
                dist_cfg.get("rungs", [base // 2, base // 4])]
    out = []
    for enc, dec in grid:
        enc = None if enc in (None, base) else int(enc)
        dec = int(dec)
        out.append((f"e{base if enc is None else enc}d{dec}",
                    enc, dec))
    return out


def variant_flops(bb: nn.Module, B: int, T: int, H: int, W: int) -> dict:
    """CNN encoder / bottleneck / decoder GFLOPs for one [B, T] forward."""
    buckets, _ = attach_flops_counter(bb)
    with torch.no_grad():
        xb = torch.randn(B, T, 1, H, W, device=next(bb.parameters()).device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            bb(xb, output_all_frames=True)
    enc_keys = ("enc1", "enc2", "enc3")
    dec_keys = DEC_STAGES
    return {
        "enc_GFLOP": sum(buckets.get(k, 0) for k in enc_keys) / 1e9,
        "bot_GFLOP": buckets.get("bottleneck", 0.0) / 1e9,
        "dec_GFLOP": sum(buckets.get(k, 0) for k in dec_keys) / 1e9,
    }


def static_radar(cfg, device: str, dist_cfg) -> dict:
    base = int(cfg.model.base_channels)
    H = W = int(cfg.data.target_resolution[0])
    T = int(cfg.data.temporal_window)
    variants = [(f"e{base}d{base}", None, base)] + variant_grid(dist_cfg, base)
    out = {}
    print(f"\n=== static radar (gpu={device}, T={T}, {H}x{W}) ===")
    print(f"{'variant':>8} {'params(M)':>10} {'encGF':>8} {'botGF':>7} "
          f"{'decGF':>8} {'cnnGF':>8} {'dec_ms':>8} {'stream_ms':>9} {'stream_fps':>9}")
    for tag, enc_base, dec_base in variants:
        model = build_student(cfg, dec_base, device, enc_base=enc_base).eval()
        freeze_bn(model)
        n_params = sum(p.numel() for p in model.backbone.parameters()) / 1e6
        fl = variant_flops(model.backbone, 1, T, H, W)
        dec_ms = decoder_wall_time_ms(model.backbone, T, H, W)
        str_ms = streaming_latency_ms(model.backbone, H, W)
        cnn_total = fl["enc_GFLOP"] + fl["bot_GFLOP"] + fl["dec_GFLOP"]
        out[tag] = {
            "enc_base": base if enc_base is None else int(enc_base),
            "dec_base": int(dec_base),
            "params_M": round(n_params, 4),
            "encoder_GFLOP_T5": round(fl["enc_GFLOP"], 3),
            "bottleneck_GFLOP_T5": round(fl["bot_GFLOP"], 3),
            "decoder_GFLOP_T5": round(fl["dec_GFLOP"], 3),
            "cnn_total_GFLOP_T5": round(cnn_total, 3),
            "decoder_wall_ms_T5": round(dec_ms, 3),
            "streaming_ms_perframe": round(str_ms, 3),
            "streaming_fps": round(1000.0 / str_ms, 1),
        }
        print(f"{tag:>8} {n_params:>10.3f} {fl['enc_GFLOP']:>8.2f} "
              f"{fl['bot_GFLOP']:>7.2f} {fl['dec_GFLOP']:>8.2f} "
              f"{cnn_total:>8.2f} {dec_ms:>8.3f} "
              f"{str_ms:>9.3f} {1000.0 / str_ms:>9.1f}")
        del model
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Agreement metric (val/agreement_teacher, identical to train_gazer.validate)
# ---------------------------------------------------------------------------
@torch.no_grad()
def agreement_val(model, loader, teacher, cfg, device, max_batches=None):
    model.eval()
    freeze_bn(model)
    C = cfg.model.num_classes
    H, W = cfg.data.target_resolution
    total, n = 0.0, 0
    torch.cuda.set_device(device)
    for batch in loader:
        images = batch["images"].to(device)
        gt = batch["gt"].to(device)
        has_gt = batch["has_gt"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model.backbone(images, output_all_frames=True).float()
        ref = teacher.get_window(images, batch["native"], gt, has_gt,
                                 batch["keys"])            # [B,T,C,H,W] fp32
        B_, T_ = logits.shape[:2]
        agree = dice_agreement(logits.reshape(-1, C, H, W),
                               ref.reshape(-1, C, H, W)).view(B_, T_)
        total += float(agree.mean())
        n += 1
        if max_batches is not None and n >= max_batches:
            break
    model.train()
    freeze_bn(model)
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Dice proxy metric: dice(student softmax, policy softmax) on val — the
# SAM3-free training-stage metric (progress toward the KL anchor). The RL
# reward itself (dice against the SAM3 landscape) is measured by the gate.
# ---------------------------------------------------------------------------
@torch.no_grad()
def dice_proxy_val(model, policy, loader, cfg, device, max_batches=None):
    model.eval()
    freeze_bn(model)
    C = cfg.model.num_classes
    H, W = cfg.data.target_resolution
    total, n = 0.0, 0
    torch.cuda.set_device(device)
    for batch in loader:
        images = batch["images"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            s_logits = model.backbone(images, output_all_frames=True).float()
            t_logits = policy.backbone(images, output_all_frames=True).float()
        B_, T_ = s_logits.shape[:2]
        # dice_agreement softmaxes `pred` and treats `teacher` as a soft
        # probability map (sums to 1 over C) -> softmax the policy logits.
        t_probs = F.softmax(t_logits.reshape(-1, C, H, W), dim=1)
        agree = dice_agreement(s_logits.reshape(-1, C, H, W),
                               t_probs).view(B_, T_)
        total += float(agree.mean())
        n += 1
        if max_batches is not None and n >= max_batches:
            break
    model.train()
    freeze_bn(model)
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Per-rung distillation (annealed)
# ---------------------------------------------------------------------------
ENC_STAGES = ("enc1", "enc2", "enc3")


def distill_rung(cfg, dist_cfg, policy: VivimGazer, val_loader,
                 train_loader, device: str, dec_base: int, tag: str,
                 logf, enc_base: int = None, init_ckpt: str = None) -> dict:
    """Annealed distillation of one (enc_base, dec_base) variant.

    Stage A (SAM3-free): the loss is KL(p_s || p_t) + alpha_tv*TV — the pure
    "annealing" objective. p_t is the frozen RL-tuned policy (gazer_ep9) on
    the same GPU; the student inherits its temporal backbone and the
    untouched CNN stages. No SAM3, no reward landscape (RAM: the GRPO train
    run holds the 118GB landscape; this run holds none). The RL reward
    (val/agreement_teacher vs the SAM3 landscape) is measured at the GATE
    stage (--gate) on the same val landscape as gazer_ep9.

    Freeze policy (target selection is by the gate's measured degradation
    tolerance, not by static FLOPs ranking):
      - temporal_mamba + GAZER head: ALWAYS frozen (the learned temporal
        capacity and the RL policy head — the model must stay anchored to
        the RL policy it anneals from).
      - enc stages that kept the policy's shape (enc_base=None): frozen
        (inherited, identical to the policy).
      - narrow enc/dec stages + skip adapters: TRAINED. The warm-start
        chain (init_ckpt = previous rung's best) makes the annealing a
        gradual fade-out: each rung starts from the already-annealed
        previous rung, so capacity is reduced step by step instead of in
        one cut."""
    m = cfg.model
    C = m.num_classes
    H, W = cfg.data.target_resolution
    T = cfg.data.temporal_window
    base = m.base_channels

    student = build_student(cfg, dec_base, device, enc_base=enc_base,
                            init_ckpt=init_ckpt)
    n_params = sum(p.numel() for p in student.backbone.parameters()) / 1e6
    n_adapt = sum(p.numel() for p in student.backbone.skip_adapters.parameters()) \
        if student.backbone.skip_adapters is not None else 0

    bb = student.backbone
    enc_changed = enc_base is not None and int(enc_base) != base
    dec_changed = int(dec_base) != base
    frozen_names = {"temporal_mamba", "gazer", "gazer_head"}
    train_names = set()
    if dec_changed:
        train_names |= set(DEC_STAGES)
    if enc_changed:
        train_names |= set(ENC_STAGES)
        # the bottleneck's INPUT width changed with enc_base -> its
        # (policy) weights don't load; it is freshly initialized and must
        # be trained. Its OUTPUT stays base_channels*8 (the Mamba3 input).
        train_names.add("bottleneck")
    if bb.skip_adapters is not None:
        train_names.add("skip_adapters")
    for p in bb.parameters():
        p.requires_grad_(False)
    for name, mod in bb.named_children():
        if name in train_names:
            for p in mod.parameters():
                p.requires_grad_(True)
    n_train = sum(p.numel() for p in bb.parameters() if p.requires_grad) / 1e6
    frozen_desc = "encoder inherited (enc1/2/3 frozen)" if not enc_changed \
        else "encoder narrow (enc1/2/3 TRAINED, bottleneck output kept)"
    print(f"[distill:{tag}] trainable={n_train:.3f}M of {n_params:.3f}M "
          f"({frozen_desc}; decoder narrow: {dec_changed}; "
          f"temporal_mamba + gazer head frozen)", flush=True)
    logf.write(f"[distill:{tag}] trainable={n_train:.3f}M (enc_changed="
               f"{enc_changed}, dec_changed={dec_changed})\n")

    opt = torch.optim.AdamW([p for p in bb.parameters() if p.requires_grad],
                            lr=dist_cfg.get("lr", 5e-4),
                            weight_decay=dist_cfg.get("weight_decay", 1e-4))
    alpha_ce = float(dist_cfg.get("alpha_ce", 1.0))
    alpha_kl = float(dist_cfg.get("alpha_kl", 1.0))
    tv0 = float(dist_cfg.get("tv_anneal_from", 1e-3))
    tv1 = float(dist_cfg.get("tv_anneal_to", 0.05))
    dloss = DistillLoss(alpha_ce=alpha_ce, alpha_kl=alpha_kl, alpha_tv=tv0)
    epochs = int(dist_cfg.get("epochs", 30))
    val_every = int(dist_cfg.get("val_every", 3))
    max_steps = int(dist_cfg.get("max_train_steps", 0) or 0)
    max_val = int(dist_cfg.get("max_val_batches", 0) or 0)

    torch.cuda.set_device(device)
    init_desc = os.path.basename(init_ckpt) if init_ckpt else "gazer_ep9 (policy)"
    print(f"[distill:{tag}] enc_base={enc_base or base} dec_base={dec_base} "
          f"params={n_params:.3f}M (skip adapters {n_adapt / 1e6:.3f}M) "
          f"init={init_desc} epochs={epochs} lr={dist_cfg.get('lr', 5e-4)}",
          flush=True)
    logf.write(f"[distill:{tag}] start enc_base={enc_base or base} "
               f"dec_base={dec_base} params={n_params:.3f}M init={init_desc}\n")

    best = {"score": -1.0, "epoch": 0}
    for epoch in range(1, epochs + 1):
        student.train()
        freeze_bn(student)
        dloss.alpha_tv = tv0 + (tv1 - tv0) * min(1.0, (epoch - 1) / max(1, epochs - 1))
        run = {}
        for i, batch in enumerate(train_loader):
            if max_steps and i >= max_steps:
                break
            images = batch["images"].to(device)                 # [B,T,1,H,W]
            B, T_ = images.shape[:2]
            N = B * T_

            # RL-tuned policy (frozen, no-grad) logits = the KL anchor p_t
            with torch.no_grad():
                torch.cuda.set_device(device)
                t_logits = policy.backbone(images, output_all_frames=True).float()
                logp_t = F.log_softmax(t_logits / float(dist_cfg.get("tau", 1.0)), dim=2)

            torch.cuda.set_device(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_logits = student.backbone(images, output_all_frames=True)
            s_logits = s_logits.float()
            logp_s = F.log_softmax(s_logits, dim=2)
            loss = dloss(logp_s.reshape(N, C, H, W),
                         logp_t.reshape(N, C, H, W),
                         None,                       # SAM3-free stage: no CE
                         s_logits.reshape(N, C, H, W))
            opt.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(
                [p for p in student.backbone.parameters() if p.grad is not None],
                float(dist_cfg.get("max_grad_norm", 1.0)))
            opt.step()
            with torch.no_grad():
                terms = dloss.terms(logp_s.detach(), logp_t.detach(), None,
                                    s_logits.reshape(N, C, H, W))
                run["loss"] = float(loss.item())
                run["kl"] = terms["kl"]
                run["tv"] = terms["tv"]
                run["grad"] = float(gn.item()) if gn is not None and bool(torch.isfinite(gn)) else 0.0

        for k, v in run.items():
            print(f"[distill:{tag}] ep{epoch} {k}={v:.4f}", flush=True)
            logf.write(f"[distill:{tag}] ep{epoch} {k}={v:.4f}\n")

        if epoch % val_every == 0 or epoch == epochs:
            score = dice_proxy_val(student, policy, val_loader, cfg, device,
                                   max_batches=max_val or None)
            print(f"[distill:{tag}] ep{epoch} VAL dice_proxy(policy)={score:.4f} "
                  f"(annealing progress)", flush=True)
            logf.write(f"[distill:{tag}] ep{epoch} VAL dice_proxy(policy)="
                       f"{score:.4f}\n")
            if score > best["score"]:
                best = {"score": score, "epoch": epoch}
                torch.save({"dec_base": dec_base,
                            "enc_base": (enc_base if enc_base is not None
                                         else base),
                            "init_ckpt": init_ckpt or GAZER_BEST,
                            "epoch": epoch,
                            "model_state_dict": student.backbone.state_dict(),
                            "val": {"dice_proxy_policy": score},
                            "params_M": n_params,
                            "config": cfg.to_dict()},
                           os.path.join(OUT_DIR, f"student_{tag}_best.pth"))
                logf.flush()
    print(f"[distill:{tag}] DONE best ep{best['epoch']} "
          f"dice_proxy(policy)={best['score']:.4f}", flush=True)
    logf.write(f"[distill:{tag}] DONE best ep{best['epoch']} "
               f"dice_proxy_policy={best['score']:.4f}\n")
    return best


# ---------------------------------------------------------------------------
# Gate stage: RL-reward gate on the val SAM3 landscape (same landscape for
# gazer_ep9 and every student) — the user-chosen objective ("RL 보상 유지
# 전제 최대 경량화"): keep val/agreement_teacher within -0.01 of the full
# model. The landscape is val-only (~16,800 frames, ~27GB RAM) so it
# coexists with the in-flight GRPO run's train landscape.
# ---------------------------------------------------------------------------
def _newest_distill_dir():
    import glob
    dirs = glob.glob(os.path.join(PROJECT_ROOT, "nnUNet_results",
                                  "gazer_distill_*"))
    return max(dirs, key=os.path.getmtime) if dirs else None


def run_gate(cfg, dist_cfg, args, policy, val_loader, dev, logf):
    """val-only SAM3 landscape + agreement of gazer_ep9 and the Stage A
    students on the SAME landscape; gate = delta >= -gate."""
    from sam.teacher import SAM3Teacher, GTTeacher
    from train_gazer import precompute_reward_landscape, GazerTeacherRouter

    device_ref = "cuda:1"
    # ref_model = EMA reference of gazer_ep9 (the eye-box source the GRPO
    # landscape was built from) — mirrors train_gazer exactly.
    ref_model = build_model(cfg).to(device_ref)
    _ck = torch.load(GAZER_BEST, map_location="cpu", weights_only=False)
    _ref_sd = {k[len("backbone."):]: v
               for k, v in _ck.get("ref_state_dict", {}).items()
               if k.startswith("backbone.")}
    if _ref_sd:
        ref_model.backbone.load_state_dict(_ref_sd, strict=True)
        print("[gate] ref_model = EMA reference of gazer_ep9 (eye-box source)",
              flush=True)
    else:
        print("[gate] WARNING: no ref_state_dict; falling back to semantic "
              "init for the eye boxes", flush=True)
    ref_model.eval()
    freeze_bn(ref_model)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    sam3 = SAM3Teacher(prompts=cfg.teacher.prompts,
                       cache_dir=cfg.teacher.cache_dir, device=device_ref,
                       num_classes=cfg.model.num_classes,
                       frame_shape=tuple(cfg.data.target_resolution),
                       use_geom=bool(getattr(cfg.teacher, "use_geom", True)))
    gt = GTTeacher(cfg.model.num_classes, tuple(cfg.data.target_resolution))

    gate_dir = args.gate_dir or _newest_distill_dir()
    if not gate_dir or not os.path.isdir(gate_dir):
        raise SystemExit("[gate] no --gate-dir with student_*_best.pth found")
    print(f"[gate] gate dir: {gate_dir}", flush=True)

    print("[gate] Phase 0 reward landscape (val only) ...", flush=True)
    t0 = time.time()
    cache = precompute_reward_landscape(cfg, args.config,
                                        val_loader.dataset, None, logf)
    print(f"[gate] landscape ready in {(time.time() - t0) / 60:.1f} min",
          flush=True)

    router = GazerTeacherRouter(cfg, sam3, gt, device_ref, ref_model)
    router.preload_cache(cache)
    gate = float(dist_cfg.get("gate", 0.01))

    print("[gate] policy (gazer_ep9) val agreement ...", flush=True)
    teacher_score = agreement_val(policy, val_loader, router, cfg, dev)
    print(f"[gate] gazer_ep9 val/agreement_teacher={teacher_score:.4f}",
          flush=True)

    base = int(cfg.model.base_channels)
    results = {"gazer_ep9": {"agreement_teacher": teacher_score}}
    for fname in sorted(os.listdir(gate_dir)):
        if not (fname.startswith("student_") and fname.endswith("_best.pth")):
            continue
        ck = torch.load(os.path.join(gate_dir, fname),
                        map_location="cpu", weights_only=False)
        # ckpt-driven shapes: legacy rungs saved only dec_base (encoder
        # stayed full width); grid variants save enc_base too.
        dec_base = int(ck["dec_base"])
        enc_raw = ck.get("enc_base", None)
        enc_base = None if (enc_raw is None or int(enc_raw) >= base) \
            else int(enc_raw)
        vtag = (fname[len("student_"):-len("_best.pth")])
        model = build_student(cfg, dec_base, dev, enc_base=enc_base)
        model.backbone.load_state_dict(ck["model_state_dict"], strict=True)
        score = agreement_val(model, val_loader, router, cfg, dev)
        delta = score - teacher_score
        passed = delta >= -gate
        results[vtag] = {"agreement_teacher": score,
                         "enc_base": base if enc_base is None else enc_base,
                         "dec_base": dec_base,
                         "params_M": ck.get("params_M"),
                         "delta_vs_teacher": delta, "gate": gate,
                         "passed": passed, "ep": ck.get("epoch")}
        print(f"[gate] {vtag}: val/agreement_teacher={score:.4f} "
              f"(delta={delta:+.4f}, gate={'PASS' if passed else 'FAIL'})",
              flush=True)
        logf.write(f"[gate] {vtag} agreement={score:.4f} delta={delta:+.4f} "
                   f"gate={'PASS' if passed else 'FAIL'}\n")
        del model
        torch.cuda.empty_cache()

    passed = [k for k, v in results.items() if k != "gazer_ep9" and v["passed"]]
    # selection = the user-chosen objective ("RL 보상 유지 전제 최대
    # 경량화"): among the gate-passing variants, the lightest one
    # (fewest params); the gate, not a static FLOPs ranking, decides
    # which variants survive.
    best_passed = min(passed, key=lambda k: (results[k].get("params_M") or 1e30),
                      default=None)
    summary = {"config": args.config, "gate_dir": gate_dir,
               "teacher": teacher_score, "gate": gate, "results": results,
               "passed": passed,
               "selected": best_passed,
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(OUT_DIR, "gate.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[gate] DONE -> {os.path.join(OUT_DIR, 'gate.json')}")
    print(f"[gate] gazer_ep9={teacher_score:.4f} | "
          + " | ".join(f"{k}={v['agreement_teacher']:.4f}({v['delta_vs_teacher']:+.4f})"
                       for k, v in results.items() if k != "gazer_ep9")
          + f"\n[gate] selected={best_passed or 'NONE (all FAIL)'}")


# ---------------------------------------------------------------------------
def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--distill-config", type=str,
                    default="configs/gazer_distill_t5.yaml")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--static", action="store_true",
                    help="static radar only (params + decoder FLOPs/time), no landscape")
    ap.add_argument("--epochs", type=int, default=None, help="override distill epochs")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run: 1-grid 2 variants, few steps")
    ap.add_argument("--gate", action="store_true",
                    help="RL-reward gate: val-only SAM3 landscape (~27GB) + "
                         "val/agreement_teacher of gazer_ep9 and every "
                         "student_*_best.pth in --gate-dir, on the SAME "
                         "landscape; gate = delta >= -gate; selection = "
                         "lightest passing variant (min params)")
    ap.add_argument("--gate-dir", type=str, default=None,
                    help="Stage A output dir holding student_*_best.pth "
                         "(default: the newest gazer_distill_* dir)")
    ap.add_argument("--start-tag", type=str, default=None,
                    help="chain resume: skip grid variants before this tag "
                         "(e.g. --start-tag e32d8 runs only e32d8/e16d16/e16d8; "
                         "the skipped rungs' best ckpts must be copied into the "
                         "new OUT_DIR first)")
    ap.add_argument("--init-ckpt", type=str, default=None,
                    help="init checkpoint for the FIRST run variant "
                         "(default: the previous variant's best in this "
                         "OUT_DIR, or gazer_ep9 for the first variant)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dist_cfg = load_config(os.path.join(PROJECT_ROOT, args.distill_config))
    set_seed(cfg.experiment.seed)
    torch.cuda.set_device(args.gpu)
    dev = f"cuda:{args.gpu}"
    H = W = int(cfg.data.target_resolution[0])
    T = int(cfg.data.temporal_window)

    OUT_DIR = os.path.join(PROJECT_ROOT, "nnUNet_results",
                           "gazer_distill_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, "distill.log"), "a")

    # ------------------------------------------------------------------
    # Static radar (no landscape needed)
    # ------------------------------------------------------------------
    static = static_radar(cfg, dev, dist_cfg)
    with open(os.path.join(OUT_DIR, "static_radar.json"), "w") as f:
        json.dump({"config": args.config, "gpu": args.gpu, **static}, f, indent=2)
    print(f"[distill] static radar -> {os.path.join(OUT_DIR, 'static_radar.json')}",
          flush=True)
    if args.static:
        logf.close()
        return

    # ------------------------------------------------------------------
    # Stage A (default / --smoke): SAM3-FREE annealed distillation.
    #   loss = KL(p_s || p_t) + alpha_tv*TV, p_t = the frozen RL-tuned
    #   policy (gazer_ep9). No SAM3, no reward landscape: the GRPO train
    #   run holds the 118GB train landscape in RAM, so the two coexist.
    #   The RL reward (val/agreement_teacher vs the SAM3 landscape) is
    #   measured at the GATE stage (--gate) on the same val landscape.
    # ------------------------------------------------------------------
    print("[distill] building RL-tuned policy (gazer_ep9) ...", flush=True)
    policy = build_teacher(cfg, dev)
    loaders = get_sequence_gazer_dataloaders(cfg)
    val_loader = loaders["validation"]

    if args.gate:
        run_gate(cfg, dist_cfg, args, policy, val_loader, dev, logf)
        logf.close()
        return

    train_loader = loaders["train"]
    if args.epochs is not None:
        dist_cfg["epochs"] = args.epochs
    base = int(cfg.model.base_channels)
    if args.smoke:
        dist_cfg["val_every"] = 1
        dist_cfg["max_train_steps"] = int(dist_cfg.get("smoke_steps", 4))
        dist_cfg["max_val_batches"] = int(dist_cfg.get("smoke_val_batches", 6))
        grid = [(None, base // 2), (base // 2, base // 2)]
    else:
        grid = variant_grid(dist_cfg, base)
        if args.start_tag:
            i0 = next((i for i, (t, _, _) in enumerate(grid)
                       if t == args.start_tag), None)
            if i0 is None:
                raise SystemExit(f"[distill] --start-tag {args.start_tag} "
                                  f"not in grid {[t for t, _, _ in grid]}")
            print(f"[distill] chain resume: skipping "
                  f"{[t for t, _, _ in grid[:i0]]}, starting at "
                  f"{grid[i0][0]}", flush=True)
            grid = grid[i0:]

    # Warm-start chain = gradual capacity fade-out: each variant starts
    # from the PREVIOUS variant's best checkpoint (shape-aware load), so
    # the annealing reduces capacity step by step instead of in one cut.
    # The first variant starts from the RL-tuned policy (gazer_ep9).
    results = {}
    prev_best = args.init_ckpt
    for tag, enc_base, dec_base in grid:
        init_ckpt = prev_best
        res = distill_rung(cfg, dist_cfg, policy, val_loader,
                           train_loader, dev, dec_base, tag, logf,
                           enc_base=enc_base, init_ckpt=init_ckpt)
        results[tag] = res
        prev_best = os.path.join(OUT_DIR, f"student_{tag}_best.pth")

    summary = {
        "config": args.config,
        "distill_config": args.distill_config,
        "stage": "A (SAM3-free KL+TV annealed distillation, warm-start chain)",
        "policy": {"checkpoint": GAZER_BEST},
        "static": static,
        "grid": [[tag, (base if e is None else int(e)), int(d)]
                 for tag, e, d in grid],
        "variants": results,
        "note": "gate = the RL reward gate; run --gate after Stage A to "
                "measure val/agreement_teacher on the val SAM3 landscape",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUT_DIR, "distill.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logf.close()

    print("\n=== distill Stage A summary (warm-start chain) ===")
    for tag, res in results.items():
        print(f"  {tag:>7}  best ep{res['epoch']:<3} "
              f"dice_proxy(policy)={res['score']:.4f}")
    print(f"\n[distill] Stage A DONE -> {os.path.join(OUT_DIR, 'distill.json')}")
    print("[distill] next: run with --gate to measure the RL reward "
          "(val/agreement_teacher) on the val SAM3 landscape")


if __name__ == "__main__":
    main()
