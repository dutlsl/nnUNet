"""
GAZER Training: on-policy GRPO tuning of the segmentation backbone
(AutoGaze stage-2 mapping; plan §7).

Policy pi_theta = the segmentation BACKBONE (Mamba3 Vivim), initialized from a
semantic fold_1 checkpoint and TRAINED (backbone unfrozen). Per-frame action =
a full segmentation mask m_t^g ~ Cat(p_t) (G=8 samples; all G rollouts share
ONE backbone forward, so VRAM ~= a single forward with grad).

Reward = per-pixel DENSE teacher-likelihood of the sampled class:
    r_t^g(x) = alpha * ref_t[x, m_t^g(x)]
  ref_t = GT one-hot (has_gt frames) else LIVE SAM3 multi-prompt soft4.
  The SAM3 reference is a fixed SIMULATOR (reward landscape), RAM-cached
  because the per-frame reference is a run constant — it is NEVER a static
  supervision target (plan §7.0).

Algorithm: GRPO (group z-score advantage, discounted returns) + exact
per-pixel KL anchor to an EMA reference model (model_ref, semantic init,
frozen/EMA, on GPU 1 — also the eye-box source). NTP 1-stage is OFF
(ntp_epochs = 0): the has_gt == 0 oracle degenerates.

GPUs: 0 = policy backbone (grad) + optimizer; 1 = SAM3 teacher + model_ref
(forwards, no-grad).

Usage:
    python train_gazer.py --config configs/gazer_grpo_t5.yaml
"""

import os
import sys
import copy
import glob
import time
import argparse
import threading
import queue
import multiprocessing as mp
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# PyTorch compat shim (mamba_ssm import chain)
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = None
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = None

from utils.config import load_config, save_config
from utils.seed import set_seed
from datasets.sequence_label_dataset import get_sequence_gazer_dataloaders
from models.vivim_gazer import VivimGazer
from rl.grpo import (categorical_logp_dense, dense_pixel_advantage,
                     compute_group_advantage, dice_agreement)
from losses.grpo_loss import GRPOLoss, pixel_kl
from sam.teacher import SAM3Teacher, GTTeacher


# ---------------------------------------------------------------------------
def build_model(cfg) -> VivimGazer:
    m = cfg.model
    model = VivimGazer(
        in_channels=1,
        num_classes=m.num_classes,
        base_channels=m.base_channels,
        d_state=m.mamba.d_state,
        d_conv=m.mamba.d_conv,
        expand=m.mamba.expand,
        mamba_headdim=m.mamba.headdim,
        use_mamba=cfg.ablation.use_mamba,
        mamba_version=cfg.ablation.mamba_version,
        temporal_window=cfg.data.temporal_window,
        gazer_d_z=m.gazer.d_z,
        gazer_n_transformer_layers=m.gazer.n_transformer_layers,
        gazer_pool_size=m.gazer.pool_size,
        gazer_hidden=m.gazer.hidden,
        use_gazer=cfg.ablation.use_gazer,
    )
    ckpt_path = getattr(m, "pretrained_checkpoint", None)
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck.get("network_weights", ck)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[GAZER] loaded pretrained backbone from {ckpt_path}")
        print(f"[GAZER]   missing (new): {len(missing)}, unexpected: {len(unexpected)}")
    else:
        print(f"[GAZER] WARNING: pretrained checkpoint not found: {ckpt_path}")
    return model


def build_teacher(cfg, device_ref: str, model_ref):
    """SAM3 (GPU1) + GT teacher + RAM reward-landscape cache; eye boxes come
    from model_ref (semantic init, on device_ref) or GT (plan §7.2 step 2)."""
    prompts = cfg.teacher.prompts
    num_classes = cfg.model.num_classes
    frame_shape = tuple(cfg.data.target_resolution)
    sam3 = None
    try:
        sam3 = SAM3Teacher(prompts=prompts,
                           cache_dir=cfg.teacher.cache_dir,
                           device=device_ref,
                           num_classes=num_classes,
                           frame_shape=frame_shape,
                           use_geom=bool(getattr(cfg.teacher, "use_geom", True)))
        if not sam3.available:
            print("[GAZER] SAM3 built but unavailable (no checkpoint); GT-only teacher.")
            sam3 = None
    except Exception as e:
        print(f"[GAZER] SAM3 unavailable ({type(e).__name__}); GT-only teacher.")
        sam3 = None
    gt = GTTeacher(num_classes, frame_shape)
    return GazerTeacherRouter(cfg, sam3, gt, device_ref, model_ref)


class GazerTeacherRouter:
    """Per-unique-frame teacher reference with an in-memory RAM cache.

    The reference is a fixed REWARD LANDSCAPE (simulator), not a supervision
    target: it is consumed by the policy-gradient reward at every step, and
    it is RAM-cached purely for speed (fp16, whole set in 251GB RAM).

    Multi-prompt eye sourcing (plan §7.2 step 2), in student 448-padded space:
      1. has_gt -> GT union bbox + center of classes {1,2,3}
      2. else   -> model_ref self-segmentation union bbox + center, computed
         with the RITnet-PREPROCESSED input (the backbone was trained on the
         preprocessed stream; feeding raw uint8 yields diffuse segmentation
         and garbage boxes — plan §7.4)
      3. empty  -> medoid box (crop [0,310,229,393] + 24px pad)
    The point prompt is a ~8px positive box at the eye center (SAM3's image
    model ignores point prompts natively). SAM3 consumes the RAW native frame.
    """

    # medoid bbox/center in 448-padded space (crop stats + 24px zero-pad)
    MEDOID_BOX = (24.0, 334.0, 253.0, 417.0)
    MEDOID_CXY = (114.0, 382.0)

    def __init__(self, cfg, sam3, gt, device_ref: str, model_ref):
        self.cfg = cfg
        self.sam3 = sam3
        self.gt = gt
        self.device = device_ref                      # e.g. cuda:1 (model_ref + SAM3)
        self.device_idx = int(device_ref.split(":")[-1])
        self.policy_device_idx = 0                    # policy backbone device (restored after ref forwards)
        self.model = model_ref
        self.amp_dtype = (torch.bfloat16
                          if getattr(cfg.training, "precision", "bf16") == "bf16"
                          else torch.float16)
        self.num_classes = cfg.model.num_classes
        self.frame_shape = tuple(cfg.data.target_resolution)
        # RAM reward-landscape cache: fp16 CPU tensors [C, H, W] keyed by frame id.
        # Whole set (73,800 frames) ~= 118GB < 251GB RAM (plan §7.3 item 1).
        self._cache: dict = {}
        self._max_frames = int(getattr(cfg.teacher, "max_cache_frames", 73800))
        self._zero = torch.zeros((self.num_classes, *self.frame_shape),
                                 dtype=torch.float16)
        self._miss = 0

    # ------------------------------------------------------------------
    def _point_box(self, cxy, half: float = 4.0, H: int = 448, W: int = 448):
        cx, cy = cxy
        return (max(0.0, cx - half), max(0.0, cy - half),
                min(float(W), cx + half), min(float(H), cy + half))

    def _bbox_from_mask(self, fg: torch.Tensor):
        """(eye_box, point_box) from a foreground map [H, W] (float)."""
        if float(fg.max()) <= 0:
            return None
        ys, xs = torch.where(fg > 0.05 * float(fg.max()))
        if len(ys) == 0:
            return None
        x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
        cxy = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        return (x0, y0, x1, y1), self._point_box(cxy)

    def _eye_boxes(self, images_frame: torch.Tensor, gt_frame: torch.Tensor,
                   has_gt: bool):
        """(eye_box, point_box) in 448-padded frame pixels (GT priority).

        images_frame: [1, H, W] float RITnet-preprocessed (for the model_ref
        self-segmentation forward on device_ref; expanded to [1,1,1,H,W]).
        """
        if has_gt:
            lab = gt_frame.cpu().numpy()
            ys, xs = np.where((lab >= 1) & (lab <= 3))
            if len(ys) > 0:
                box = (float(xs.min()), float(ys.min()),
                       float(xs.max()), float(ys.max()))
                cxy = ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0)
                return box, self._point_box(cxy)
        if self.model is not None:
            torch.cuda.set_device(self.device_idx)
            try:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    x = images_frame.to(self.device)
                    if x.dim() == 3:                         # [1,H,W] -> [1,1,1,H,W]
                        x = x.view(1, 1, 1, *x.shape[1:])
                    logits = self.model.backbone(x, output_all_frames=True).float()
            finally:
                torch.cuda.set_device(self.policy_device_idx)
            probs = F.softmax(logits, dim=2)                 # [1,1,4,448,448]
            fg = probs[:, :, 1:].sum(dim=2).squeeze()        # [448,448]
            bbox = self._bbox_from_mask(fg)
            if bbox is not None:
                return bbox
        return self.MEDOID_BOX, self._point_box(self.MEDOID_CXY)

    # ------------------------------------------------------------------
    def _call(self, images_frame: torch.Tensor, native_frame: np.ndarray,
              gt_frame: torch.Tensor, has_gt: bool) -> torch.Tensor:
        """Returns CPU fp16 soft teacher [C, H, W]."""
        if has_gt:
            return self.gt(gt_frame).float().to(torch.float16).cpu()
        native = native_frame.astype(np.uint8) if native_frame.dtype != np.uint8 else native_frame
        soft = torch.zeros((self.num_classes, *self.frame_shape), dtype=torch.float16)
        if self.sam3 is None:
            return soft
        eye_box, point_box = self._eye_boxes(images_frame, gt_frame, False)
        soft4, _conf = self.sam3(native, eye_box=eye_box, point_box=point_box)
        return soft4.squeeze(0).to(torch.float16).cpu()

    def get_window(self, images: torch.Tensor, native: np.ndarray,
                   gt: torch.Tensor, has_gt: torch.Tensor,
                   keys: list) -> torch.Tensor:
        """images [B,T,1,H,W] float (preprocessed, policy device) — for the
        model_ref eye-box forward; native [B,T,H0,W0] uint8 — for SAM3 only.
        Returns soft teacher [B, T, C, H, W] float32 on images' device."""
        B, T = images.shape[:2]
        H, W = images.shape[-2:]
        out = []
        for b in range(B):
            for t in range(T):
                key = str(keys[b * T + t])
                cached = self._cache.get(key)
                if cached is None:
                    self._miss += 1
                    if not bool(has_gt[b, t]) and self.sam3 is None:
                        self._cache[key] = self._zero
                    else:
                        self._cache[key] = self._call(images[b, t], native[b, t],
                                                      gt[b, t], bool(has_gt[b, t]))
                    if len(self._cache) > self._max_frames:
                        self._cache.pop(next(iter(self._cache)))
                    cached = self._cache[key]
                out.append(cached)
        soft = torch.stack(out, dim=0)                       # [B*T, C, H, W] fp16
        return soft.float().to(images.device).view(B, T, self.num_classes, H, W)

    def preload_cache(self, cache: dict):
        """Load the Phase-0 reward-landscape cache (key -> fp16 [C,H,W])."""
        for k, v in cache.items():
            self._cache[str(k)] = v

    def clear(self):
        self._cache.clear()

    @property
    def cache_size(self):
        return len(self._cache)

    @property
    def cache_misses(self):
        return self._miss


# ---------------------------------------------------------------------------
# Phase 0: precompute the SAM3 reward landscape on ALL GPUs in parallel.
#
# The per-frame teacher reference is a run-constant SIMULATOR (SAM3 is
# deterministic, the semantic-init eye-box model is frozen). Filling the RAM
# reward-landscape cache ahead of training (a) keeps it RAM-only (no disk
# supervision dataset — plan §7.0) and (b) removes the serial SAM3-on-GPU1
# bottleneck that idles the policy GPU ~65% during epoch 1. Two worker
# processes (one per GPU; Mamba3's Triton kernels are single-GPU-per-process)
# each build their own model_ref + SAM3, compute per-frame soft4, and stream
# fp16 frames back to the parent's in-RAM cache. Phase 1 (GRPO) then runs
# with a warm cache: zero SAM3 calls, policy GPU busy the whole time.
# ---------------------------------------------------------------------------
def _collect_frames(ds) -> dict:
    """key (int frame number) -> (img_path, has_gt, semantic_label_root).
    Merged from a dataset's windows; a frame maps to a single entry."""
    frames = {}
    for win, has_gt in ds.samples:
        for fpath, hg in zip(win, has_gt):
            key = int(os.path.splitext(os.path.basename(fpath))[0])
            frames[key] = (fpath, bool(hg), ds.semantic_label_root)
    return frames


def _phase0_worker(gpu_idx: int, cfg_path: str, items: list,
                   result_q: "mp.Queue", err_q: "mp.Queue",
                   ref_batch: int = 16):
    """One Phase 0 worker on a dedicated GPU.

    items: list of (key, img_path, has_gt, label_root).
    For each frame: read png -> crop [400x400] -> RITnet-preprocess (ref input)
    + raw pad (SAM3 input) -> model_ref self-seg (T=1 batched) -> eye box ->
    SAM3 soft4 (or GT one-hot if has_gt). Streams (key, soft4 fp16) back.
    """
    import torch
    import cv2
    import numpy as np
    from utils.config import load_config
    from utils.seed import set_seed
    from train_gazer import build_model, _point_box_static
    from sam.teacher import SAM3Teacher
    from datasets.openeds_dataset import RITnetPreprocessor

    cfg = load_config(cfg_path)
    set_seed(cfg.experiment.seed)
    dev = f"cuda:{gpu_idx}"
    torch.cuda.set_device(gpu_idx)
    H = W = int(cfg.data.target_resolution[0])
    y0, y1, x0, x1 = tuple(cfg.data.input_crop)
    crop_h, crop_w = y1 - y0, x1 - x0
    pad = (H - crop_h) // 2                             # symmetric pad (24)
    num_classes = cfg.model.num_classes

    model_ref = build_model(cfg).to(dev)
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)
    teacher_sam3 = SAM3Teacher(
        prompts=cfg.teacher.prompts, cache_dir=cfg.teacher.cache_dir,
        device=dev, num_classes=num_classes,
        frame_shape=(H, W), use_geom=bool(getattr(cfg.teacher, "use_geom", True)))
    if not teacher_sam3.available:
        err_q.put(f"[phase0 GPU{gpu_idx}] FATAL: SAM3 unavailable")
        return
    prep = RITnetPreprocessor(target_size=(crop_h, crop_w))

    done = 0
    for start in range(0, len(items), ref_batch):
        chunk = items[start:start + ref_batch]
        ref_imgs, natives, meta = [], [], []
        for (key, img_path, has_gt, label_root) in chunk:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((640, 400), dtype=np.uint8)
            raw = img[y0:y1, x0:x1]                     # [crop_h, crop_w]
            natives.append(np.pad(raw, ((pad, pad), (pad, pad))))
            pre = np.pad(prep(raw).astype(np.float32), ((pad, pad), (pad, pad)))
            ref_imgs.append(pre)
            meta.append((key, has_gt, label_root))
        ref_t = torch.from_numpy(
            np.ascontiguousarray(np.stack(ref_imgs)[None]))        # [1,B,H,W]
        ref_t = ref_t.view(1, len(chunk), 1, H, W).to(dev).float()

        # Batched self-segmentation on the preprocessed input (T=1 per frame —
        # identical to the live per-frame path; gate-verified 0.961 fg Dice).
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model_ref.backbone(ref_t, output_all_frames=True).float()
        probs = torch.softmax(logits, dim=2)                       # [1,B,4,H,W]
        fg = probs[:, :, 1:].sum(dim=2).squeeze(0).cpu().numpy()   # [B,H,W]
        eye_boxes = []
        for bi in range(len(chunk)):
            fgb = fg[bi]
            # NOTE: must not be a ternary — `a = X if c else Y, Z` binds the
            # conditional to the first element only, so xs would be empty.
            if fgb.max() > 0:
                ys, xs = np.where(fgb > 0.05 * fgb.max())
            else:
                ys, xs = np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)
            if len(ys) == 0:
                eye_boxes.append((GazerTeacherRouter.MEDOID_BOX,
                                  _point_box_static(GazerTeacherRouter.MEDOID_CXY)))
                continue
            box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            cxy = ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0)
            eye_boxes.append((box, _point_box_static(cxy)))

        for bi, (key, has_gt, label_root) in enumerate(meta):
            if has_gt and label_root:
                lbl = np.load(os.path.join(label_root, f"{key}.npy")).astype(np.int64)
                lbl448 = np.pad(lbl[y0:y1, x0:x1], ((pad, pad), (pad, pad)))
                soft = np.zeros((num_classes, H, W), dtype=np.float16)
                for c in range(num_classes):
                    soft[c] = (lbl448 == c).astype(np.float16)
            else:
                soft4, _ = teacher_sam3(natives[bi], eye_box=eye_boxes[bi][0],
                                        point_box=eye_boxes[bi][1])
                soft = soft4.squeeze(0).to(torch.float16).cpu().numpy()
            result_q.put((int(key), soft.astype(np.float16)))
            done += 1
            if done % 200 == 0:
                print(f"[phase0 GPU{gpu_idx}] {done}/{len(items)} frames "
                      f"({done / len(items) * 100:.0f}%)", flush=True)
    result_q.put(("__DONE__", gpu_idx, done))


def _point_box_static(cxy, half: float = 4.0, H: int = 448, W: int = 448):
    cx, cy = cxy
    return (max(0.0, cx - half), max(0.0, cy - half),
            min(float(W), cx + half), min(float(H), cy + half))


def precompute_reward_landscape(cfg, cfg_path: str, train_ds, val_ds,
                                logf) -> dict:
    """Phase 0: fill the in-RAM reward-landscape cache on all GPUs.

    Returns {key_str: soft4 fp16 tensor [C,H,W]} to hand to the teacher router.
    The per-frame reference is the fixed SIMULATOR (SAM3 deterministic +
    frozen semantic-init eye boxes); computing it in a warm-up phase removes
    the serial SAM3 bottleneck that idles the policy GPU during epoch 1.
    RAM-only — nothing touches disk (plan §7.0).
    """
    num_gpus = torch.cuda.device_count()
    merged = _collect_frames(train_ds)
    if val_ds is not None:
        merged.update(_collect_frames(val_ds))      # train/val frame ids disjoint
    keys = sorted(merged)
    n = len(keys)
    print(f"[phase0] {n} unique frames across {num_gpus} GPUs "
          f"(est ~{n / num_gpus * 0.85 / 3600:.1f}h at ~0.85 s/frame/GPU)",
          flush=True)
    logf.write(f"[phase0] {n} unique frames, {num_gpus} GPUs\n"); logf.flush()

    items = [(int(k),) + merged[k] for k in keys]
    halves = [items[i::num_gpus] for i in range(num_gpus)]
    ctx = mp.get_context("spawn")
    result_q, err_q = ctx.Queue(), ctx.Queue()
    procs = []
    for gi in range(num_gpus):
        p = ctx.Process(target=_phase0_worker,
                        args=(gi, cfg_path, halves[gi], result_q, err_q,
                              int(getattr(cfg.teacher, "ref_batch", 16))))
        p.start()
        procs.append(p)

    cache = {}
    expected = n
    done_workers = set()   # gpu_idx whose __DONE__ arrived
    t0 = time.time()
    while len(cache) < expected:
        # Timeout get() instead of a bare blocking get(): if a worker dies
        # mid-run its __DONE__ never arrives and a blocking get() would hang
        # the process forever (alive but idle -> watchdog would not relaunch).
        try:
            item = result_q.get(timeout=1.0)
        except queue.Empty:
            for gi, p in enumerate(procs):
                if p.is_alive():
                    continue
                # A cleanly-finished worker (exitcode==0, __DONE__ seen) is
                # NOT a crash: it delivered its assigned chunk and the
                # remaining frames come from the other worker(s). The
                # finished-first worker vs. the still-working one is a
                # normal race at phase0 end — keep draining the queue.
                if p.exitcode == 0 and gi in done_workers:
                    continue
                while not err_q.empty():
                    print(err_q.get(), flush=True)
                p.join()
                raise RuntimeError(
                    f"phase0 GPU{gi} worker died (exitcode={p.exitcode}) "
                    f"at {len(cache)}/{expected} frames "
                    f"(__DONE__={gi in done_workers})")
            continue
        if item[0] == "__DONE__":
            done_workers.add(item[1])
            continue
        key, soft = item
        cache[str(key)] = torch.from_numpy(soft)
        if len(cache) % 2000 == 0:
            rate = len(cache) / max(1e-6, time.time() - t0)
            eta_h = (expected - len(cache)) / max(1e-6, rate) / 3600
            print(f"[phase0] cache {len(cache)}/{expected} "
                  f"({rate:.1f} frames/s, ETA {eta_h:.1f}h)", flush=True)
            logf.write(f"[phase0] cache {len(cache)}/{expected} "
                       f"(ETA {eta_h:.1f}h)\n"); logf.flush()
    # surface worker errors
    while not err_q.empty():
        e = err_q.get()
        print(e, flush=True); logf.write(e + "\n"); logf.flush()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"phase0 worker exited with code {p.exitcode}")
    if len(cache) < expected:
        raise RuntimeError(f"phase0 incomplete: {len(cache)}/{expected} frames")
    gb = len(cache) * cfg.model.num_classes * cfg.data.target_resolution[0] \
        * cfg.data.target_resolution[1] * 2 / 1e9
    print(f"[phase0] reward-landscape cache complete: {len(cache)} frames "
          f"(~{gb:.0f}GB RAM, {time.time() - t0:.0f}s)", flush=True)
    logf.write(f"[phase0] complete: {len(cache)} frames\n"); logf.flush()
    return cache


def grpo_step(model, model_ref, batch, teacher, cfg, device, device_ref_idx,
              amp_dtype, grpo_loss_fn: GRPOLoss, epoch) -> dict:
    """One on-policy GRPO step over the segmentation backbone (plan §7.2).

    DENSE per-pixel variant (2026-10-27): the reward, advantage, and
    importance ratio are all per-pixel — no pixel-mean collapse.

    1) logits [B,T,C,H,W] -> logp = log_softmax(logits/tau)
    2)+(3) ref = teacher.get_window  (eye boxes from model_ref/GT, SAM3 on native)
    4) m^g ~ Cat(p_t) (G samples, ONE backbone forward shared);
       logpi^g(x) = logp_t(x, m^g(x)) per pixel;  logpi_ref^g from model_ref
    5) r^g(x) = alpha * ref_t[x, m^g(x)]  (+ beta * TC, off in config)
    6) A^g(x) = clip(r^g(x) - mean_g r^g(x), +-adv_clip)
       (group-centered RAW; no std norm — confident pixels get A=0 exactly)
    7) L = -mean exp(logpi^g - logpi_ref^g) * A^g
          + beta_kl * mean KL( pi_theta || pi_ref )  (exact per-pixel KL)
       grad -> backbone only.
    """
    images = batch["images"].to(device)          # [B,T,1,H,W]
    gt = batch["gt"].to(device)                  # [B,T,H,W]
    has_gt = batch["has_gt"].to(device)          # [B,T]
    native = batch["native"]
    keys = batch["keys"]
    B, T = images.shape[0], images.shape[1]
    C, H, W = cfg.model.num_classes, images.shape[-2], images.shape[-1]
    N = B * T
    G = cfg.grpo.group_size
    tau = cfg.grpo.temperature

    # 1) Policy forward WITH grad — the single forward shared by all G rollouts.
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        logits = model.backbone(images, output_all_frames=True)   # [B,T,C,H,W]
    logp = F.log_softmax(logits.float() / tau, dim=2)

    # 2)+(3) Live teacher reference (RAM-cached reward landscape).
    ref = teacher.get_window(images, native, gt, has_gt, keys)    # [B,T,C,H,W] fp32

    # 4) Reference policy logp (model_ref on GPU1, no-grad) — KL anchor +
    #    importance-ratio baseline.
    dev_ref = f"cuda:{device_ref_idx}"
    torch.cuda.set_device(device_ref_idx)
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
            logits_ref = model_ref.backbone(images.to(dev_ref), output_all_frames=True)
    finally:
        torch.cuda.set_device(device)
    logp_ref = F.log_softmax(logits_ref.float() / tau, dim=2).detach().to(device)
    del logits_ref

    logp_f = logp.view(N, C, H, W)
    logp_ref_f = logp_ref.view(N, C, H, W)
    ref_f = ref.view(N, C, H, W)

    # Sample G masks (categorical, pixel-wise; Gumbel internally) and gather
    # per-pixel log-probs and rewards DENSELY — no pixel-mean anywhere.
    # The old frame-scalar path collapsed the dense per-pixel reward to one
    # number per frame over ~2e5 pixels; with the near-deterministic semantic
    # init the G rollouts' frame rewards differed by ~1.6e-4, so the group
    # z-score (std floor 1e-2) was 60x above the real variance and the
    # advantage was pure floor noise (policy gradient ~ 0 from ep2 on).
    with torch.no_grad():
        p_flat = logp_f.permute(0, 2, 3, 1).reshape(N * H * W, C)  # [N*H*W, C]
        m = torch.distributions.Categorical(logits=p_flat).sample((G,))
        m = m.view(G, N, H, W)                                    # [G,N,H,W]

    # 4)+(5) Dense per-pixel teacher-likelihood reward, [G,N,H,W] (detached
    # landscape; grad would flow only through the ratio below).
    ref_pix = ref_f.permute(0, 2, 3, 1)                           # [N,H,W,C]
    rewards_pix = cfg.grpo.alpha * ref_pix.unsqueeze(0).expand(G, N, H, W, C) \
        .gather(4, m.unsqueeze(-1)).squeeze(-1).detach()          # [G,N,H,W]

    # 6)+(7) Dense group-centered advantage (raw centering, no std norm —
    # confident pixels where all G agree get A = 0 exactly).
    A_dense = dense_pixel_advantage(rewards_pix, adv_clip=cfg.grpo.adv_clip)

    # Per-pixel log-probs of each sampled mask (grad through logp only).
    logp_dense = categorical_logp_dense(logp_f, m)                # [G,N,H,W]
    logp_ref_dense = categorical_logp_dense(logp_ref_f, m)        # [G,N,H,W]
    del m, ref_pix

    # 8) Loss: dense policy gradient + exact per-pixel KL anchor (grad ->
    # backbone only; KL uses the full categorical dists, per-frame).
    kl = pixel_kl(logp_f, logp_ref_f)                             # [N], grad
    loss = grpo_loss_fn.dense(logp_dense, logp_ref_dense, A_dense, kl)
    loss.backward()

    with torch.no_grad():
        neg_pg = float(-(torch.exp((logp_dense - logp_ref_dense).clamp(-20, 20))
                        * A_dense).mean().item())
        kl_val = GRPOLoss.kl_divergence(logp_f, logp_ref_f)
        entropy = float(-(logp.exp() * logp).sum(dim=2).mean().item())
    return {
        "grpo/loss": float(loss.item()),
        "grpo/neg_pg": neg_pg,
        "grpo/kl": kl_val,
        "grpo/adv_std": float(A_dense.std().item()),
        "grpo/reward": float(rewards_pix.mean().item()),
        "grpo/entropy": entropy,
        "gazer/teacher_cache": float(teacher.cache_size),
        "gazer/teacher_miss": float(teacher.cache_misses),
    }


# ---------------------------------------------------------------------------
def freeze_bn(model: VivimGazer):
    """Pin every BatchNorm layer to eval mode (frozen running stats).

    The GRPO policy is fine-tuned from the semantic fold_1 init; its BN
    running stats are that init's statistics. In train mode BN would
    recompute per-batch stats and drift its running buffers, while
    model_ref (EMA of `parameters()` only — buffers are NOT EMA'd) would
    keep the init stats: the KL anchor would then measure the BN-stat
    mismatch (~2 nats) instead of the true weight drift, breaking the
    plan §7.2 invariants (init KL ~0.03, importance ratio = 1). Both
    policy and ref must share the same (frozen) BN statistics.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def set_train_split(model: VivimGazer):
    """Restore train mode + clear the FIFO temporal cache.

    Called at each epoch boundary and after validate(), so stale
    validation frames cached in the backbone FIFO do not leak into the
    first training batch of the next epoch. BN stays frozen (freeze_bn)
    — see freeze_bn docstring.
    """
    model.train()
    freeze_bn(model)
    model.backbone.reset_cache()


def update_ema(model_ref: VivimGazer, model: VivimGazer, tau: float):
    """EMA the reference BACKBONE (KL anchor + eye-box source).

    Buffers (BN running stats) are EMA'd too. freeze_bn pins the policy's
    buffers, so they should not drift — this keeps model_ref an honest
    EMA of the WHOLE backbone if that ever changes.
    """
    with torch.no_grad():
        for p_ref, p in zip(model_ref.backbone.parameters(), model.backbone.parameters()):
            p_ref.data.mul_(tau).add_(p.data.to(p_ref.device), alpha=1.0 - tau)
        for b_ref, b in zip(model_ref.backbone.buffers(), model.backbone.buffers()):
            if b_ref.dtype.is_floating_point:
                b_ref.data.mul_(tau).add_(b.data.to(b_ref.device), alpha=1.0 - tau)
        if model_ref.gazer is not None and model.gazer is not None:
            for p_ref, p in zip(model_ref.gazer.parameters(), model.gazer.parameters()):
                p_ref.data.mul_(tau).add_(p.data.to(p_ref.device), alpha=1.0 - tau)


def validate(model, loader, teacher, cfg, device, amp_dtype) -> dict:
    """Validation: Dice on GT frames (if any) + agreement with the LIVE teacher
    (SAM3 soft4 on no-GT frames) — the primary checkpoint score (plan §7.3 7)."""
    model.eval()
    metrics = {}
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            gt = batch["gt"].to(device)
            has_gt = batch["has_gt"].to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model.backbone(images, output_all_frames=True)
            logits = logits.float()
            probs = F.softmax(logits, dim=2)
            pred = probs.argmax(2)                       # [B,T,H,W]
            ref = teacher.get_window(images, batch["native"], gt, has_gt,
                                     batch["keys"])      # [B,T,C,H,W] fp32
            B_, T_ = logits.shape[:2]
            # Dice on GT frames (rare/absent in Sequence: has_gt ~ 0)
            mask = has_gt.bool()
            if mask.any():
                oh = F.one_hot(gt[mask], cfg.model.num_classes).permute(0, 3, 1, 2).float()
                pm = pred[mask].float()
                inter = (pm * oh).sum((2, 3))
                dice = (2 * inter + 1e-5) / (pm.sum((2, 3)) + oh.sum((2, 3)) + 1e-5)
                metrics["val/dice_gt"] = metrics.get("val/dice_gt", 0.0) + float(dice.mean())
                metrics["val/n_gt_frames"] = metrics.get("val/n_gt_frames", 0.0) \
                    + float(mask.sum())
            # Agreement with the live teacher (all frames; GT one-hot or SAM3)
            agree = dice_agreement(
                logits.reshape(-1, *logits.shape[2:]),
                ref.reshape(-1, *ref.shape[2:]),
            ).view(B_, T_)
            metrics["val/agreement_teacher"] = metrics.get("val/agreement_teacher", 0.0) \
                + float(agree.mean())
            total += 1
    set_train_split(model)
    for k in list(metrics):
        if k != "val/n_gt_frames":
            metrics[k] /= max(1, total)
    metrics["val/teacher_cache"] = float(teacher.cache_size)
    return metrics


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    device_ref = (f"cuda:{(args.gpu + 1) % torch.cuda.device_count()}"
                  if torch.cuda.device_count() > 1 else device)
    device_ref_idx = int(device_ref.split(":")[-1])
    amp_dtype = torch.bfloat16 if cfg.training.precision == "bf16" else torch.float16
    # Mamba3 Triton kernels are single-GPU-per-process: pin the process's
    # current CUDA device to the policy GPU. model_ref / SAM3 forwards run on
    # device_ref and restore this in a finally (see grpo_step / _eye_boxes).
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    out_base = os.path.join(PROJECT_ROOT, "nnUNet_results")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or os.path.join(out_base, f"{cfg.experiment.name}_{ts}")
    os.makedirs(output_dir, exist_ok=True)
    save_config(cfg, os.path.join(output_dir, "config_snapshot.yaml"))
    logf = open(os.path.join(output_dir, "train.log"), "w")

    # Setup order (plan §7.3 item 0): load the semantic fold_1 checkpoint FIRST,
    # then deepcopy — model_ref must carry the semantic weights (it is the
    # eye-box source AND the KL anchor). Backbone is TRAINABLE (unfrozen);
    # model_ref is frozen + EMA.
    model = build_model(cfg).to(device)
    freeze_bn(model)
    n_bb = sum(p.numel() for p in model.backbone.parameters())
    print(f"[GAZER] policy backbone TRAINABLE ({n_bb / 1e6:.1f}M params), "
          f"use_gazer={cfg.ablation.use_gazer}")
    model_ref = copy.deepcopy(model)
    model_ref.to(device_ref)
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)

    teacher = build_teacher(cfg, device_ref, model_ref)
    max_cache = int(getattr(cfg.teacher, "max_cache_frames", 73800))
    cache_gb = max_cache * cfg.model.num_classes * (cfg.data.target_resolution[0]
                * cfg.data.target_resolution[1]) * 2 / 1e9
    print(f"[GAZER] teacher: SAM3={'on' if teacher.sam3 else 'OFF (GT-only)'} "
          f"on {device_ref}; RAM reward-landscape cache {max_cache} frames "
          f"(~{cache_gb:.0f}GB fp16)")

    loaders = get_sequence_gazer_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders.get("validation")

    # Phase 0 (plan §7.0): precompute the SAM3 reward landscape into the RAM
    # cache on both GPUs before training -> Phase 1 GRPO runs warm-cache with
    # zero live SAM3 calls (no serial teacher bottleneck on the policy GPU).
    if bool(getattr(cfg.teacher, "precompute", True)):
        landscape = precompute_reward_landscape(
            cfg, args.config, loaders["train"].dataset,
            loaders.get("validation").dataset
            if loaders.get("validation") is not None else None,
            logf)
        teacher.preload_cache(landscape)
        del landscape

    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(bb_params, lr=cfg.training.backbone_lr,
                                  weight_decay=cfg.training.weight_decay)
    ntp_epochs = int(cfg.grpo.ntp_epochs)
    grpo_epochs = int(cfg.grpo.grpo_epochs)
    total_epochs = ntp_epochs + grpo_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs))
    grpo_loss_fn = GRPOLoss(beta_kl=cfg.grpo.beta_kl, lambda_q=0.0)

    log_every = cfg.training.log_every
    best_val = -1.0
    patience_counter = 0

    for epoch in range(1, total_epochs + 1):
        stage = "ntp" if epoch <= ntp_epochs else "grpo"
        set_train_split(model)
        run = {}
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            stats = grpo_step(model, model_ref, batch, teacher, cfg, device,
                              device_ref_idx, amp_dtype, grpo_loss_fn, epoch)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in bb_params if p.grad is not None],
                cfg.training.max_grad_norm)
            optimizer.step()
            # pre-clip total grad norm = the real "policy gradient alive"
            # signal (neg_pg is structurally ~0 at init: mean_g A = 0 and
            # ratio = 1, so the PG direction shows up only here).
            if grad_norm is not None and bool(torch.isfinite(grad_norm)):
                stats["grad/norm"] = float(grad_norm.item())
            for k, v in stats.items():
                run.setdefault(k, []).append(v)

            if (i + 1) % log_every == 0:
                msg = f"ep{epoch} [{stage}] it{i+1}/{len(train_loader)} " + " ".join(
                    f"{k}={np.mean(run[k]):.4f}" for k in sorted(run))
                print(msg, flush=True)
                logf.write(msg + "\n")
                logf.flush()

        scheduler.step()

        # KL anneal for GRPO (beta_kl: 0.05 -> 0.01 over the run)
        if stage == "grpo":
            lo, hi = cfg.grpo.kl_schedule
            frac = min(1.0, (epoch - ntp_epochs) / max(1, grpo_epochs))
            beta_kl = hi + (lo - hi) * frac
            grpo_loss_fn.beta_kl = beta_kl

        # EMA reference update (backbone; KL anchor + eye-box source)
        if stage == "grpo":
            update_ema(model_ref, model, cfg.grpo.ref_ema_tau)

        # Validation (primary score: agreement with the live teacher)
        val = validate(model, val_loader, teacher, cfg, device, amp_dtype)
        val_msg = f"ep{epoch} [{stage}] VAL " + " ".join(
            f"{k}={v:.4f}" for k, v in sorted(val.items()))
        print(val_msg, flush=True)
        logf.write(val_msg + "\n")

        score = val.get("val/agreement_teacher", val.get("val/dice_gt", -1.0))
        if score > best_val:
            best_val = score
            patience_counter = 0
            torch.save({"epoch": epoch, "stage": stage, "model_state_dict":
                        model.state_dict(), "ref_state_dict":
                        model_ref.state_dict(), "val": val,
                        "config": cfg.to_dict()},
                       os.path.join(output_dir, "checkpoint_best.pth"))
            print(f"  * best {score:.4f} saved", flush=True)
            logf.write(f"  * best {score:.4f} saved\n")
            logf.flush()
        else:
            patience_counter += 1
        if epoch % 5 == 0:
            torch.save({"epoch": epoch, "stage": stage, "model_state_dict":
                        model.state_dict(), "ref_state_dict":
                        model_ref.state_dict(), "val": val,
                        "config": cfg.to_dict()},
                       os.path.join(output_dir, "checkpoint_latest.pth"))

        min_ep = int(getattr(cfg.training.early_stopping, "min_epochs", 0))
        if epoch >= min_ep and patience_counter >= cfg.training.early_stopping.patience:
            print(f"[GAZER] early stop at epoch {epoch}", flush=True)
            break

    logf.close()
    print(f"[GAZER] done. best={best_val:.4f} out={output_dir}", flush=True)


if __name__ == "__main__":
    main()
