# 🖥️ RTX 4090 시연 서버 Mamba-3 구축 및 검증 가이드 (Deployment Guide)

본 가이드는 **NVIDIA RTX 4090 시연 서버**에서 Mamba-3 (arXiv:2603.15569) 기반 192×192 RITnet 백본 모델 및 비디오 세그멘테이션 파이프라인을 부작용 없이 설치, 검증 및 시연 구동하기 위한 **표준 운영 절차서(SOP)**입니다.

---

## 📋 1. 하드웨어 및 OS 권장 사양

- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM) 1장 이상
- **CUDA / Driver**: Driver 535.xx 이상 (CUDA 12.1 / 12.2 지원)
- **OS**: Ubuntu 22.04 LTS (x86_64)
- **Python**: Python 3.10.x

---

## ⚙️ 2. 가상환경 구축 및 패키지 설치 (Step-by-Step)

### Step 2.1. Python 가상환경 생성 및 활성화
```bash
conda create -n nnunet_mamba3 python=3.10 -y
conda activate nnunet_mamba3
```

### Step 2.2. PyTorch (CUDA 12.1) 설치
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2.3. Triton 및 Mamba-3 커널 엔진 패키지 설치
Mamba-3는 1세대 `selective_scan_fn` 대신 Triton 커널(`mamba3_siso_combined`, `layernorm_gated`)을 사용합니다.
```bash
pip install triton>=3.0.0
pip install mamba-ssm==2.3.2.post1 --no-build-isolation
```

### Step 2.4. 필수 유틸리티 및 데이터셋 연동 패키지 설치
```bash
pip install opencv-python-headless pillow numpy tqdm wandb simpleitk
```

---

## 🚨 3. 필수 환경 변수 설정 (Mandatory Runtime Envs)

Mamba-3의 Triton 커널 연산은 PyTorch Dynamo (`torch.compile`)와 충돌하므로 **반드시 `nnUNet_compile=false` 환경 변수를 비활성화**해야 합니다.

터미널 실행 또는 시연 스크립트 상단에 아래 환경 변수를 추가하십시오:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
export nnUNet_compile=false
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0
```

---

## 🔍 4. Mamba-3 환경 및 가중치 무결성 자동 검증 (Verification)

서버 구축 완료 후, 제공된 **자동 검증 스크립트**를 실행하여 6단계 무결성을 점검합니다:

```bash
python agent/verify_mamba3_environment.py
```

---

## 🎯 5. 192×192 RITnet 추론 / 시연 실행 방법

학습된 `checkpoint_best.pth` 가중치를 로드하여 3-Frame 비디오 시퀀스 입력에 대해 안구 구성 요소(Sclera, Iris, Pupil)를 192×192 해상도로 실시간 세그멘테이션합니다.

```python
import torch
from models.vivim_backbone import VivimBackbone
from datasets.openeds_dataset import RITnetPreprocessor

# 1. 모델 생성
model = VivimBackbone(in_channels=1, num_classes=4, base_channels=32, d_state=16, use_mamba=True).cuda()

# 2. 가중치 로드
ckpt = torch.load("nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_Vivim__nnUNetPlans__2d/fold_1/checkpoint_best.pth", map_location="cuda")
state_dict = {k.replace("backbone.", ""): v for k, v in ckpt["network_weights"].items()}
model.load_state_dict(cleaned_state_dict, strict=False)
model.eval()

# 3. 3-Frame 비디오 시퀀스 추론 (Input: [B=1, T=3, C=1, H=192, W=192])
with torch.no_grad():
    logits = model(x_5d_sequence)
    pred = torch.argmax(logits, dim=1)  # [B, 192, 192] (Classes: 0:BG, 1:Sclera, 2:Iris, 3:Pupil)
```
