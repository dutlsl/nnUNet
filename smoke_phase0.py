"""Smoke test for Phase 0 (plan §7.0): 2-GPU spawn precompute of the RAM
reward-landscape cache on a small frame subset (~200 frames).

Reuses the exact production path: _collect_frames + _phase0_worker (spawn) +
result_q streaming + GazerTeacherRouter.preload_cache. Validates:
  - both GPUs work in parallel (one worker each)
  - streaming queue round-trips (cache count == frame count)
  - fp16 [C,H,W] payload sanity (shape/dtype/finite)
  - preload_cache path
  - rate + extrapolated full-set ETA
RAM-only; writes nothing to disk.
"""
import os
import sys
import time
import queue
import resource
import multiprocessing as mp

import torch

from utils.config import load_config
from utils.seed import set_seed
from datasets.sequence_label_dataset import get_sequence_gazer_dataloaders
from train_gazer import (build_model, build_teacher, _collect_frames,
                         _phase0_worker)

N_SMOKE = 200
CFG_PATH = "configs/gazer_grpo_t5.yaml"


def main():
    cfg = load_config(CFG_PATH)
    set_seed(cfg.experiment.seed)
    loaders = get_sequence_gazer_dataloaders(cfg)
    frames = _collect_frames(loaders["train"].dataset)
    val_loader = loaders.get("validation")
    if val_loader is not None:
        frames.update(_collect_frames(val_loader.dataset))
    keys = sorted(frames)[:N_SMOKE]
    items = [(int(k),) + frames[k] for k in keys]
    n = len(keys)

    num_gpus = torch.cuda.device_count()
    assert num_gpus == 2, f"expected 2 GPUs, got {num_gpus}"
    halves = [items[i::num_gpus] for i in range(num_gpus)]
    ctx = mp.get_context("spawn")
    result_q, err_q = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(
        target=_phase0_worker,
        args=(gi, CFG_PATH, halves[gi], result_q, err_q,
              int(getattr(cfg.teacher, "ref_batch", 16))))
        for gi in range(num_gpus)]
    for p in procs:
        p.start()

    cache = {}
    t0 = time.time()
    while len(cache) < n:
        try:
            item = result_q.get(timeout=1.0)
        except queue.Empty:
            for gi, p in enumerate(procs):
                if not p.is_alive():
                    while not err_q.empty():
                        print(err_q.get(), flush=True)
                    raise RuntimeError(
                        f"worker GPU{gi} died (exitcode={p.exitcode}) "
                        f"at {len(cache)}/{n} frames")
            continue
        if item[0] == "__DONE__":
            continue
        key, soft = item
        cache[str(key)] = torch.from_numpy(soft)

    dt = time.time() - t0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            while not err_q.empty():
                print(err_q.get(), flush=True)
            raise RuntimeError(f"worker exitcode={p.exitcode}")

    k0, v0 = next(iter(cache.items()))
    v0f = v0.float()
    print(f"[smoke] cache {len(cache)}/{n} frames in {dt:.1f}s "
          f"= {len(cache) / dt / num_gpus:.2f} frames/s/GPU")
    print(f"[smoke] sample: key={k0} shape={tuple(v0.shape)} dtype={v0.dtype} "
          f"finite={bool(torch.isfinite(v0f).all())} "
          f"min={v0f.min().item():.4f} max={v0f.max().item():.4f} "
          f"mean_fg_mass={float(v0f.sum(dim=1).mean().item()):.2f}")
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[smoke] parent RSS ~{rss_gb:.1f}GB after cache fill")

    # preload path (after workers exit so both GPUs are free)
    model_ref = build_model(cfg).to("cuda:1")
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)
    teacher = build_teacher(cfg, "cuda:1", model_ref)
    teacher.preload_cache(cache)
    print(f"[smoke] preload_cache ok: cache_size={teacher.cache_size}")

    total_frames = len(frames)
    est_h = total_frames / (len(cache) / dt) / 3600
    print(f"[smoke] PASS: 2-GPU spawn + result_q streaming + preload. "
          f"Extrapolated full-set ({total_frames} frames) ETA: {est_h:.1f}h")
    sys.exit(0)


if __name__ == "__main__":
    main()
