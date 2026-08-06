# Mamba-3 Integration and SAGD Metric Zero-Plateau Incident Analysis Report

**Date**: 2026-08-06  
**Target Architecture**: Vivim + SAGD (SAS + HDM + SGA) with Official Mamba-3 (`mamba_ssm.modules.mamba3.Mamba3`)  
**Environment**: PyTorch 2.5.1 + CUDA 11.8 / Dual RTX 3090  

---

## 1. Incident Overview

During the training run of `nnUNetTrainer_Vivim_SADG` using the official Mamba-3 backbone on GPU 0 (`PID 1049934`), the training loss stabilized around `0.49`, but the validation Dice score dropped to `0.0000` starting from Epoch 25 and remained stuck at `0.0000` through Epoch 52 ("Zero-Plateau Collapse").

This report documents:
1. Verification of Mamba-3 integration from the `backbone` branch (`remotes/dutlsl/backbone`).
2. PyTorch 2.5 runtime Float8 compatibility resolution.
3. Pretrained weight loading verification.
4. Root cause diagnosis of the Dice zero-plateau phenomenon.
5. Actionable technical recommendations.

---

## 2. Official Mamba-3 Integration Verification

### 2.1 Branch Sync and Module Verification
- **Pulled Branch**: `dutlsl/backbone` (`commit: edbab78`)
- **Module File**: [models/mamba_block.py](file:///home/iulab1/PycharmProjects/nnUNet/models/mamba_block.py)
- **Module Interface**:
  ```python
  from mamba_ssm.modules.mamba3 import Mamba3

  class MambaLayer(nn.Module):
      def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2, headdim: int = 64, ngroups: int = 1, chunk_size: int = 64, **kwargs):
          super().__init__()
          self.mamba3 = Mamba3(
              d_model=d_model,
              d_state=d_state,
              expand=expand,
              headdim=headdim,
              ngroups=ngroups,
              chunk_size=chunk_size,
          )

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          return self.mamba3(x)
  ```

### 2.2 PyTorch 2.5 Runtime Compatibility Polyfill
During initialization on PyTorch 2.5.1, `mamba_ssm.modules.mamba3` imports `quack.cute_dsl_utils`, which references `torch.float8_e8m0fnu` (introduced natively in PyTorch 2.6+). This caused an `AttributeError`.

**Resolution**: Added a non-destructive PyTorch Float8 attribute polyfill prior to importing Mamba3 in [models/mamba_block.py](file:///home/iulab1/PycharmProjects/nnUNet/models/mamba_block.py):
```python
import torch
import torch.nn as nn

# Polyfill PyTorch float8 dtypes required by Cutlass/Quack in PyTorch 2.5
for dt in ['float8_e8m0fnu', 'float4_e2m1fn_x2', 'float8_e4m3fnuz', 'float8_e5m2fnuz']:
    if not hasattr(torch, dt):
        setattr(torch, dt, getattr(torch, 'float8_e5m2', torch.float32))

from mamba_ssm.modules.mamba3 import Mamba3
```
- **Validation**: Full pipeline smoke test (`tests/test_sadg_full.py`) passed all CUDA forward/backward checks on GPU 1.

---

## 3. Pretrained Weight Loading Verification

Verification of state dict loading from `checkpoint_final.pth` into `VivimSAGDWrapper`:

| Parameter Group | Baseline Checkpoint Keys | Matched & Loaded Keys | Status |
| :--- | :---: | :---: | :--- |
| **Encoder (`enc1` ~ `enc4`)** | 32 | 32 | **100% Loaded** |
| **Decoder (`dec1` ~ `dec4`)** | 32 | 32 | **100% Loaded** |
| **Bottleneck (`bottleneck`)** | 8 | 8 | **100% Loaded** |
| **Segmentation Head (`seg_outputs`)** | 8 | 8 | **100% Loaded** |
| **Input Stem (`stem`)** | 16 | 16 | **100% Loaded** |
| **1st-Gen Mamba Scan Keys** | 9 | 0 | *Excluded (Replaced by Mamba3)* |
| **Total Checkpoint Weights** | **105** | **96** | **96/105 Loaded (91.4%)** |

> [!NOTE]
> All 96 core Vivim baseline backbone weights (Encoder, Decoder, Bottleneck, Segmentation Head) loaded with 100% key and tensor shape alignment.
> The 9 unmapped keys belonged to the deprecated 1st-generation `selective_scan` implementation. The 142 "missing" keys in the log belong to newly added SAGD sub-modules (`SAS`, `HDM`, `SGA`, and official `Mamba3` fused projection layers).

---

## 4. Root Cause Analysis: Validation Dice Zero-Plateau

### 4.1 Trajectory Log Analysis
- **Epoch 0 ~ 4**: Pseudo Dice fluctuated between `0.1008` and `0.2144` (Sclera Dice `0.4974`).
- **Epoch 8**: Sclera Dice `0.2272`, Iris Dice `0.1668`, Mean `0.1314`.
- **Epoch 24**: Iris Dice `0.5782`, Mean `0.1986`.
- **Epoch 25 ~ 52**: Dice Mean dropped to `0.0000`, Precision = `0.0000`, Recall = `0.0000`.

### 4.2 Mechanism of the Collapse
1. **Uninitialized Mamba3 Projection Layers**: While the Vivim CNN backbone loaded pretrained weights, official Mamba-3's inner state projection layers (`in_proj`, `out_proj`, `x_proj`) were randomly initialized.
2. **Dominant Background Class**: In OpenEDS ocular images, >90% of spatial pixels belong to Class 0 (Background).
3. **Local Minimum Trapping**: Around Epoch 25, under combined multi-domain loss pressure ($L_{total} = L_{DiceCE} + 0.1 L_{DC} + 0.05 L_{SC}$), the uncalibrated Mamba3 representations pushed the segmentation head into predicting Class 0 (Background) for all spatial locations.
4. **Zero-Gradient Barrier**: When all predictions collapse to Class 0:
   $$\text{True Positives (TP)} = 0 \implies \text{Dice} = 0.0000, \quad \text{Precision} = 0.0000, \quad \text{Recall} = 0.0000$$
   Because cross-entropy loss on background pixels reaches a low value (~0.49), gradient signals for foreground classes become extremely small, trapping the network in a zero-foreground local minimum.

---

## 5. Technical Recommendations & Action Plan

To resolve the zero-plateau and ensure stable convergence with official Mamba-3:

1. **Mamba-3 & SAGD Domain Loss Warmup (Recommended)**:
   - Apply a 10-epoch linear warmup schedule to domain generalisation loss weights:
     $$w_{dc}(t) = \min\left(0.1, \frac{t}{10} \times 0.1\right), \quad w_{sc}(t) = \min\left(0.05, \frac{t}{10} \times 0.05\right)$$
   - This allows the Mamba-3 projection layers to adapt to the segmentation task before domain consistency forces alignment.

2. **Foreground Class-Weighted DiceCE Loss**:
   - Increase Cross-Entropy weight ($ce\_weight: 0.5$) or add foreground class weighting ($[0.1, 1.0, 1.0, 1.0]$) to penalize background-only collapse.

3. **Checkpoints & Git Push Status**:
   - All fixes (`models/mamba_block.py` polyfill, `openeds_dataset.py`, `nnUNetTrainer_Vivim_SADG.py` path resolution) have been committed and pushed to `dutlsl/sadg` (`commit: dcc03ac`).
