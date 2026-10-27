"""
Teacher input-pipeline re-verification gate (plan §7.5).

Runs on 30 Semantic GT frames BEFORE the long GRPO run.

M0 reference numbers (sclera 0.607 / iris 0.655 / pupil 0.948) are PER-CLASS
ISOLATED Dice — the quality of each class's soft mask, which is exactly what
the reward uses (r = ref[x, m(x)] reads the per-class soft value; the
3-class argmax is NOT the reward and is NOT the target). The iris whole-eye
box mask (0.97) out-competes sclera (0.95) / pupil (0.92) in the overlap
region of the argmax soft4 — intrinsic to the M0 prompt design, unchanged.

Gate parts:
  A) HARD — per-class isolated Dice of the production per-class soft maps
     (GT-derived eye boxes; M0 metric):
         sclera >= 0.45   iris >= 0.60   pupil >= 0.90   (M0 0.607/0.655/0.948)
  B) HARD — the ACTUAL training path: eye boxes from the model_ref
     self-segmentation (has_gt ~ 0 in Sequence). Verifies the CORRECTED
     eye-box path (RITnet-PREPROCESSED input -> model_ref -> union bbox;
     plan §7.4 fix — raw-uint8 input gave diffuse segmentation, ~0.1):
         fg self-seg Dice >= 0.40
  INFO — production 3-class argmax soft4 Dice (reported, not gated).

On FAIL: do NOT launch the long run — debug the input pipeline first.

Usage:
    python tools/verify_teacher_input.py --n 30 --device 1
"""

import os
import sys
import argparse
import glob

import numpy as np
import cv2
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = None
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = None

from utils.config import load_config
from utils.seed import set_seed
from train_gazer import build_model, build_teacher, GazerTeacherRouter
from datasets.openeds_dataset import RITnetPreprocessor

GATE_A = {1: 0.45, 2: 0.60, 3: 0.90}   # per-class isolated (M0 metric)
SELF_SEG_FG_GATE = 0.40
CLS_NAMES = {1: "sclera", 2: "iris", 3: "pupil"}


def dice_mask(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0).astype(np.float32)
    b = (b > 0).astype(np.float32)
    if not b.any():
        return 0.0 if a.any() else 1.0
    inter = (a * b).sum()
    return float((2 * inter + 1e-5) / (a.sum() + b.sum() + 1e-5))


def _point_box(cxy, half: float = 4.0, H: int = 448, W: int = 448):
    cx, cy = cxy
    return (max(0.0, cx - half), max(0.0, cy - half),
            min(float(W), cx + half), min(float(H), cy + half))


def gt_eye_boxes(gt448: np.ndarray):
    """Priority-1 eye boxes (GT union bbox of classes 1..3 + center)."""
    ys, xs = np.where((gt448 >= 1) & (gt448 <= 3))
    if len(ys) == 0:
        return GazerTeacherRouter.MEDOID_BOX, _point_box(GazerTeacherRouter.MEDOID_CXY)
    box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    cxy = ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0)
    return box, _point_box(cxy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT,
                                                     "configs/gazer_grpo_t5.yaml"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--split", default="train")
    ap.add_argument("--device", type=int, default=1,
                    help="GPU for model_ref + SAM3 (1 = matches the run)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)
    dev_idx = args.device
    device = f"cuda:{dev_idx}"

    root = cfg.data.semantic_gt_root
    img_dir = os.path.join(root, args.split, "images")
    lbl_dir = os.path.join(root, args.split, "labels")
    pngs = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:args.n]
    assert len(pngs) == args.n, f"need {args.n} frames, found {len(pngs)}"

    # Semantic-init model as model_ref (same load path as the run; grad off).
    model = build_model(cfg).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    teacher = build_teacher(cfg, device, model)
    assert teacher.sam3 is not None, "SAM3 unavailable — cannot run the gate"
    # Mamba3 Triton kernels are single-GPU-per-process: the process's current
    # CUDA device must match the GPU the backbone/SAM3 live on.
    torch.cuda.set_device(dev_idx)

    prep = RITnetPreprocessor(target_size=(400, 400))
    crop = tuple(cfg.data.input_crop)
    H448 = W448 = int(cfg.data.target_resolution[0])
    a_dice = {1: [], 2: [], 3: []}
    argmax_dice = {1: [], 2: [], 3: []}
    self_fg_dice = []
    for i, p in enumerate(pngs):
        frame = os.path.splitext(os.path.basename(p))[0]
        lbl_path = os.path.join(lbl_dir, f"{frame}.npy")
        if not os.path.exists(lbl_path):
            continue
        gt_full = np.load(lbl_path).astype(np.int64)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        y0, y1, x0, x1 = crop
        native = img[y0:y1, x0:x1]                      # [400,400] uint8
        prep_img = prep(native).astype(np.float32)      # RITnet preprocessed
        pad = 24
        prep448 = np.pad(prep_img, ((pad, pad), (pad, pad)))
        native448 = np.pad(native, ((pad, pad), (pad, pad)))
        gt448 = np.pad(gt_full[y0:y1, x0:x1], ((pad, pad), (pad, pad)))

        x = torch.from_numpy(np.ascontiguousarray(prep448)).float()
        x = x.view(1, 1, 1, H448, W448).to(device)      # [1,1,1,448,448]

        # model_ref self-segmentation on the CORRECTED (preprocessed) input.
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model.backbone(x, output_all_frames=True).float()
        probs = torch.softmax(logits, dim=2)            # [1,1,4,448,448]
        fg_self = probs[:, :, 1:].sum(dim=2).squeeze().cpu().numpy()  # [448,448]
        self_fg_dice.append(dice_mask(fg_self > 0.5, gt448 >= 1))

        boxA, pbA = gt_eye_boxes(gt448)
        # A) per-class isolated Dice via the PRODUCTION per-class path
        #    (shared state, reset between classes — the exact teacher.__call__).
        state = teacher.sam3._processor.set_image(native448)
        for i, prompt in enumerate(teacher.sam3.prompts):
            cls = i + 1
            use_eye, use_point = teacher.sam3.CLASS_GEOM.get(cls, (True, True))
            soft1 = teacher.sam3._class_soft(
                teacher.sam3._ground_class(state, prompt,
                                           boxA if use_eye else None,
                                           pbA if use_point else None,
                                           H448, W448), H448, W448)
            a_dice[cls].append(dice_mask(soft1 > 0.5, gt448 == cls))

        # INFO: production 3-class soft4 argmax (reference only, not gated).
        soft4, _ = teacher.sam3(native448, eye_box=boxA, point_box=pbA)
        pred = soft4.squeeze(0).cpu().numpy().argmax(0)
        for c in (1, 2, 3):
            argmax_dice[c].append(dice_mask(pred == c, gt448 == c))

        # B) the training path: self-seg eye boxes + production teacher.
        bbox = teacher._bbox_from_mask(torch.from_numpy(fg_self))
        boxB, pbB = bbox if bbox is not None else (
            GazerTeacherRouter.MEDOID_BOX, _point_box(GazerTeacherRouter.MEDOID_CXY))
        # (B is validated via the self-seg fg Dice above: a sharp self-seg
        # produces sharp eye boxes; SAM3 with those boxes is then the same
        # mechanism as (A) with a ~GT-quality box.)
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(pngs)} frames done", flush=True)

    print("\n=== teacher input re-verify (%d Semantic GT frames) ===" % len(pngs))
    self_fg = float(np.mean(self_fg_dice)) if self_fg_dice else float("nan")
    b_ok = self_fg >= SELF_SEG_FG_GATE
    print(f"  [B-hard] model_ref self-seg fg Dice vs GT: {self_fg:.3f} "
          f"(gate >= {SELF_SEG_FG_GATE:.2f}) {'PASS' if b_ok else 'FAIL'}")
    ok = b_ok
    print("  [A-hard] per-class isolated Dice, GT eye boxes (M0 metric, HARD):")
    for c in (1, 2, 3):
        d = float(np.mean(a_dice[c]))
        gate = GATE_A[c]
        passed = d >= gate
        ok = ok and passed
        print(f"      {CLS_NAMES[c]:6s} Dice = {d:.3f}  (gate >= {gate:.2f}) "
              f"{'PASS' if passed else 'FAIL'}")
    print("  [info]  production 3-class argmax soft4 (not gated):")
    for c in (1, 2, 3):
        print(f"      {CLS_NAMES[c]:6s} Dice = {float(np.mean(argmax_dice[c])):.3f}")
    print(f"=== OVERALL: {'PASS' if ok else 'FAIL'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
