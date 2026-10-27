"""
One-shot comparison of three models on the SAME validation set with the SAME
SAM3 reward-landscape metric (val/agreement_teacher from train_gazer.validate).

Models:
  gazer_ep9      = nnUNet_results/gazer_grpo_t5_20261026_103944/checkpoint_best.pth
                   (the current GRPO run's best; its in-log VAL ep1-ep8
                   0.7764->0.7778 is reproducible here — sanity check)
  init_fold1     = semantic fold_1 init (ep0 baseline: what the GRPO run
                   started from; quantifies how much of the 0.7778 is the
                   init and how much is 9 epochs of GRPO)
  baseline_400   = pseudo-label sequence baseline
                   (vivim_mamba3_400_t5; the reference result the user
                   defined as the bar to beat)

Metric: Dice agreement between model softmax and the per-frame SAM3 soft4
reference (or GT one-hot where has_gt), mean over all val windows. This is
exactly train_gazer.validate's val/agreement_teacher.

Cost: Phase 0 landscape recompute for the val frames only (~16,800 frames,
~27GB RAM, ~2h at the measured ~2.4 frames/s on both GPUs), then 3 full val
passes (~10 min total).
RAM-only, no disk supervision data, Sequence_PseudoLabels_400 used ONLY as a
checkpoint source (model weights), never as data.

Usage:
    python tools/gazer_eval_compare.py --config configs/gazer_grpo_t5.yaml
"""
import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = None
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = None

from utils.config import load_config
from utils.seed import set_seed
from datasets.sequence_label_dataset import get_sequence_gazer_dataloaders
from models.vivim_gazer import VivimGazer
from train_gazer import (build_model, precompute_reward_landscape,
                         GazerTeacherRouter, freeze_bn)
from rl.grpo import dice_agreement

GAZER_BEST = os.path.join(
    PROJECT_ROOT, "nnUNet_results/gazer_grpo_t5_20261026_103944/checkpoint_best.pth")
BASELINE_400 = os.path.join(
    PROJECT_ROOT, "nnUNet_results/vivim_mamba3_400_t5_20260908_102205/checkpoint_best.pth")


def load_into_backbone(model: VivimGazer, ckpt_path: str):
    """Load a VivimBackbone state dict (flat or backbone.-prefixed) into
    model.backbone; report missing/unexpected."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        sd = ck["model_state_dict"]
    elif isinstance(ck, dict) and "network_weights" in ck:
        sd = ck["network_weights"]
    else:
        sd = ck
    sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    if not sd:
        sd = ck["model_state_dict"] if "model_state_dict" in ck else sd
    missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
    print(f"[eval] {os.path.basename(ckpt_path)}: missing={len(missing)} "
          f"unexpected={len(unexpected)}")
    return model


@torch.no_grad()
def eval_val(model, loader, cache, cfg, device, tag):
    """Replicate train_gazer.validate's agreement metric with a CPU cache."""
    model.eval()
    freeze_bn(model)
    H, W = cfg.data.target_resolution
    C = cfg.model.num_classes
    total_agree, n = 0.0, 0
    for batch in loader:
        images = batch["images"].to(device)
        gt = batch["gt"].to(device)
        has_gt = batch["has_gt"].to(device)
        native = batch["native"]
        keys = batch["keys"]
        B, T = images.shape[:2]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model.backbone(images, output_all_frames=True)
        logits = logits.float()
        refs = []
        for b in range(B):
            for t in range(T):
                k = str(keys[b * T + t])
                ref = cache.get(k)
                if ref is None:
                    ref = torch.zeros((C, H, W), dtype=torch.float16)
                refs.append(ref)
        ref = torch.stack(refs).view(B, T, C, H, W).float().to(device)
        agree = dice_agreement(logits.reshape(-1, C, H, W),
                               ref.reshape(-1, C, H, W)).view(B, T)
        total_agree += float(agree.mean())
        n += 1
    return total_agree / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)
    torch.cuda.set_device(args.gpu)
    dev = f"cuda:{args.gpu}"

    loaders = get_sequence_gazer_dataloaders(cfg)
    val_loader = loaders["validation"]

    # Phase 0 on the VAL split ONLY — the 3 passes are all val, and the
    # Sequence split has no GT frames (has_gt == 0 everywhere), so the
    # landscape is needed only for the val frames (~16,800; ~27GB RAM,
    # ~2h at the measured 2.4 frames/s). Building the full 73,800-frame
    # landscape here would 4.4x the cost for frames never read.
    logf = open(os.path.join(PROJECT_ROOT, "logs/diagnostics/gazer_eval_compare.log"),
                "a")
    t0 = time.time()
    print("[eval] Phase 0 reward landscape (val only) ...", flush=True)
    cache = precompute_reward_landscape(
        cfg, args.config, loaders["validation"].dataset, None, logf)
    print(f"[eval] landscape ready in {(time.time() - t0) / 60:.1f} min", flush=True)

    results = {}
    for tag, ckpt in [
        ("gazer_ep9", GAZER_BEST),
        ("init_fold1", cfg.model.pretrained_checkpoint),
        ("baseline_400", BASELINE_400),
    ]:
        model = build_model(cfg).to(dev)      # loads the semantic init
        if ckpt and os.path.exists(ckpt):
            load_into_backbone(model, ckpt)
        freeze_bn(model)
        t1 = time.time()
        score = eval_val(model, val_loader, cache, cfg, dev, tag)
        dt = time.time() - t1
        results[tag] = {"agreement_teacher": score, "val_pass_s": round(dt, 1)}
        print(f"[eval] {tag}: val/agreement_teacher={score:.4f} ({dt / 60:.1f} min)",
              flush=True)
        logf.write(f"{tag}={score:.4f} t={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        del model
        torch.cuda.empty_cache()

    out = os.path.join(PROJECT_ROOT, "logs/diagnostics/gazer_eval_compare.json")
    with open(out, "w") as f:
        json.dump({"config": args.config, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   **results}, f, indent=2)
    logf.close()
    print(f"\n[eval] DONE -> {out}")
    for tag, r in results.items():
        print(f"  {tag:14s} {r['agreement_teacher']:.4f}")


if __name__ == "__main__":
    main()
