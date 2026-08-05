# SAGD (Structure-Aware Domain Generalization) Single-GPU Execution Guide

본 가이드는 싱글 GPU 서버(RTX 3090, RTX 4090, RTX 5090 등)에서 코드를 내려받은 직후 즉시 Vivim+SAGD 학습을 구동하기 위한 통합 안내서입니다.

---

## 1. 태그 정보
- **Git Tag**: `v1.0.0-sagd-single-gpu`
- **주요 기능**: Mamba3 + Vivim + Structure-Aware Serialization (SAS) + Hierarchical Domain Modeling (HDM) + Spectral Graph Alignment (SGA) 싱글 GPU 순차 학습

---

## 2. 방법 A: 가상환경 (venv / conda) 직접 구동

### 2.1 가상환경 생성 및 패키지 설치
```bash
python3.10 -m venv nnunet_env
source nnunet_env/bin/activate

# 1. PyTorch 2.3.1 (CUDA 12.1 / CUDA 12.6 호환)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Mamba 및 Causal Conv1d (RTX 5090/Blackwell 환경에서도 nvcc 컴파일 필요)
pip install causal-conv1d==1.4.0
pip install mamba-ssm==2.2.2 --no-build-isolation

# 3. 프로젝트 의존성 설치
pip install -r requirements.txt
pip install -e .
```

### 2.2 싱글 GPU 학습 구동 명령어
```bash
source nnunet_env/bin/activate

# nnUNet 필수 환경변수 설정
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# 타겟 GPU 지정 (예: 0번 GPU)
export CUDA_VISIBLE_DEVICES=0

# 백그라운드 학습 구동 (nohup)
nohup nnUNetv2_train Dataset600_OpenEDS2019 2d 0 \
  -tr nnUNetTrainer_Vivim_SADG \
  > logs/sadg_train_single_gpu.log 2>&1 &

# 로그 실시간 확인
tail -f logs/sadg_train_single_gpu.log
```

---

## 3. 방법 B: Docker 컨테이너 구동 (환경 충돌 제로)

RTX 5090 등 새로운 GPU 아키텍처나 서버 환경 라이브러리 충돌이 우려될 경우 Docker 빌드를 권장합니다.

### 3.1 Docker 이미지 빌드
```bash
docker build -t sagd-vivim:v1.0.0 .
```

### 3.2 Docker 컨테이너 실행 및 학습
```bash
docker run --gpus '"device=0"' --ipc=host -it \
  -v /path/to/data/nnUNet_raw:/workspace/nnUNet/nnUNet_raw \
  -v /path/to/data/nnUNet_preprocessed:/workspace/nnUNet/nnUNet_preprocessed \
  -v /path/to/data/nnUNet_results:/workspace/nnUNet/nnUNet_results \
  sagd-vivim:v1.0.0 \
  nnUNetv2_train Dataset600_OpenEDS2019 2d 0 -tr nnUNetTrainer_Vivim_SADG
```

---

## 4. 무결성 검증 포인트
1. **Sequential Sampling**: `MultiDomainBatchSampler`가 시간적 연속성을 보장하도록 각 에포크 커서 방식으로 동작합니다.
2. **SAGD DG Loss**: DiceCE (`ce_weight=0.3`), Domain Consistency (`weight=0.1`), Structural Contrastive (`weight=0.05`)가 모두 활성화되어 도메인 일반화 손실을 수렴시킵니다.
3. **GPU 0 단일 점유**: `CUDA_VISIBLE_DEVICES=0` 설정 시 6GB~8GB VRAM 범위 내에서 안정적으로 구동됩니다.
