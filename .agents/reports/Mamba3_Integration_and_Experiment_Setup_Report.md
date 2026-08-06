# 🚀 Mamba-3 공식 모듈 탑재 및 192×192 RITnet 백본 학습 전수 검증 보고서

> **작성 일시**: 2026-08-06  
> **보고서 목적**: Mamba-3 공식 패키지 모듈 연동 버그 수정, 192×192 RITnet 전처리 파이프라인 전수 검증, WandB 온라인 스트리밍 연동 및 학습 속도/성능 모니터링 종합 보고  

---

## 📌 1. 개요 및 변경 사항 (Summary of Changes)

1. **Mamba-3 공식 패키지 모듈 직접 연동**:
   - `models/mamba_block.py` 내 기존 1세대 `selective_scan_fn` C++ 커널 바인딩 및 수동 프로젝션(`in_proj`, `x_proj`, `dt_proj`) 코드를 전면 제거함.
   - `mamba_ssm.modules.mamba3.Mamba3` 공식 릴리즈 클래스(arXiv:2603.15569)를 직접 인스턴스화하여 Exponential-Trapezoidal Discretization, RoPE Complex-Valued State Update, SISO/MIMO fused Triton 커널이 지원되는 최신 구조로 리팩토링 완료.

2. **`torch.compile` 비활성화 및 환경 분리**:
   - Mamba-3 Triton 커널과 PyTorch Dynamo/Inductor 간 불투명 Custom Op 충돌 방지를 위해 `run_host_backbone.sh`에 `export nnUNet_compile=false` 환경 변수 추가.

3. **WandB 온라인 스트리밍 활성화**:
   - `run_host_backbone.sh` 및 `run_host_sagd.sh`의 `export WANDB_MODE=offline`을 `export WANDB_MODE=online`으로 변경하여 WandB 대시보드 실시간 연동 완료.

---

## 📌 2. 실험 조건 14개 항목 전수 검증표 (Full Conditions Checklist)

| 분류 | 실험 조건 항목 | 세부 설정 값 / 연산 방식 | 적용 코드 / 파일 위치 | 검증 결과 |
|---|---|---|---|:---:|
| **데이터 및 해상도** | 1. 입력 이미지 해상도 | **192×192** (Pupil Core 네이티브 해상도) | [openeds_dataset.py](file:///home/iulab0/PycharmProjects/nnUNet/datasets/openeds_dataset.py#L131) | 🟢 **192×192 적용** |
| | 2. 라벨 데이터 해상도 | **192×192** (4-Class Pseudo Label) | `Sequence_PseudoLabels_192` | 🟢 **192×192 적용** |
| | 3. 시퀀스 길이 ($T$) | **$T=3$** (3프레임 연속 비디오 타임윈도우) | [openeds_dataset.py](file:///home/iulab0/PycharmProjects/nnUNet/datasets/openeds_dataset.py#L141) | 🟢 **T=3 적용** |
| **전처리 파이프라인** | 4. Gamma Correction | **$\gamma = 0.8$** Look-Up Table 적용 | [openeds_dataset.py](file:///home/iulab0/PycharmProjects/nnUNet/datasets/openeds_dataset.py#L132) | 🟢 **Gamma 0.8 적용** |
| | 5. CLAHE 평활화 | **`clipLimit=1.5`**, **`tileGridSize=(8,8)`** | [openeds_dataset.py](file:///home/iulab0/PycharmProjects/nnUNet/datasets/openeds_dataset.py#L133) | 🟢 **CLAHE 1.5 적용** |
| | 6. 픽셀 정규화 범위 | **`[-1.0, 1.0]`** Range (`(img - 0.5) / 0.5`) | [openeds_dataset.py](file:///home/iulab0/PycharmProjects/nnUNet/datasets/openeds_dataset.py#L136) | 🟢 **[-1, 1] 정규화 적용** |
| **모델 아키텍처** | 7. Mamba Engine | **Mamba-3 공식 모듈** (`mamba3.Mamba3`) | [mamba_block.py](file:///home/iulab0/PycharmProjects/nnUNet/models/mamba_block.py#L15) | 🟢 **Mamba-3 연동** |
| | 8. Spatio-Temporal Scan | Bottleneck **Temporal Mamba Block (TMB)** | [temporal_mamba.py](file:///home/iulab0/PycharmProjects/nnUNet/models/temporal_mamba.py#L13) | 🟢 **TMB 적용** |
| | 9. Mamba Hyperparameters | `d_state=16`, `d_conv=4`, `expand=2`, `headdim=64` | [nnUNetTrainer_Vivim.py](file:///home/iulab0/PycharmProjects/nnUNet/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainer_Vivim.py#L170) | 🟢 **하이퍼파라미터 적용** |
| **학습 인자 (Hparams)** | 10. Batch Size | **11** | `plans.json` (2d config) | 🟢 **Batch=11 적용** |
| | 11. Initial LR / Schedule | **`0.01`** (Polynomial LR Decay, 1000 Epochs) | `nnUNetTrainer.py` | 🟢 **LR 0.01 적용** |
| | 12. Loss Function | **Batch Dice + CrossEntropy** (`DC_and_CE_loss`) | `nnUNetTrainer.py` | 🟢 **Dice+CE 적용** |
| **실행 및 모니터링** | 13. PyTorch Compile | **`export nnUNet_compile=false`** (Triton 연동) | [run_host_backbone.sh](file:///home/iulab0/PycharmProjects/nnUNet/run_host_backbone.sh#L8) | 🟢 **false 적용** |
| | 14. WandB Sync Mode | **`export WANDB_MODE=online`** (실시간 동기화) | [run_host_backbone.sh](file:///home/iulab0/PycharmProjects/nnUNet/run_host_backbone.sh#L6) | 🟢 **Online 스트리밍 중** |

---

## 📌 3. 실시간 학습 성능 및 속도 벤치마크 (Current Status)

- **학습 실행 PID**: Background detached process (via `nohup`)
- **GPU 사용 현황**: VRAM ~8.3 GB / GPU Utilization ~99%
- **WandB 온라인 대시보드**: [https://wandb.ai/hyun02051-hongik-university/eyeball-3d/runs/beg9b9es](https://wandb.ai/hyun02051-hongik-university/eyeball-3d/runs/beg9b9es)

### 📈 학습 속도 비교 (Epoch Time)
- **이전 수동 스크립트 (1세대 커널)**: Epoch 당 **~74.6초**
- **현재 Mamba-3 공식 Triton 연동**: Epoch 당 **`24.16초`** (**약 3.1배 속도 개선 달성**)

### 📊 수렴 지표 (Epoch 68 기준)
- `train_loss`: **`-0.9328`**
- `val_loss`: **`-0.9399`**
- `Pseudo Dice`: **Sclera `0.9567`**, **Iris `0.9734`**, **Pupil `0.9694`**
- `EMA Pseudo Dice`: **`0.9271`** (지속 상승 중)

---

*본 보고서는 백본 리팩토링 및 192×192 RITnet 전처리 실험의 공식 기록으로 보존됩니다.*
