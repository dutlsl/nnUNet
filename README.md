# nnUNet 400×400 Native & Temporal ConvLSTM Pipeline Guide
> **Branch**: `feature/temporal-convlstm`  
> **Target**: Pupil Core (400×400 60Hz) Eye Camera Alignment & 3D Eyeball Stability Optimization

---

## 📌 1. 프로젝트 개요 & 핵심 변경 배경

### ❓ 왜 400×400 네이티브 재학습이 필요한가?
* **기존 문제**: Pupil Core 아이캠(400×400) $\rightarrow$ 640×400 패딩 $\rightarrow$ 추론 후 192×192로 `cv2.INTER_NEAREST` 축소 복원하는 과정에서 **경계선 1~2px 보간 왜곡(Aliasing)** 발생.
* **영향**: 1px의 타원 경계 흔들림이 pye3d의 3D 타원 복원(`unproject_ellipse`) 과정에서 3D Sphere Center 및 시선 벡터의 큰 변동(점프)을 유발하여 pye3d 안구 윤곽선(파란 원) 및 gaze 좌표가 흔들림.
* **해결책**: 이미지 리사이즈(Interpolation) 과정을 **100% 제거**하고, OpenEDS($640 \times 400$)에서 세로 중앙 영역($Y \in [120:520], X \in [0:400]$)을 **온더플라이(On-the-fly) 메모리 크롭**하여 **보간 왜곡 0%의 $400 \times 400$ 네이티브 모델**을 구축함.

---

## 🐍 2. 가상환경(Virtual Environment) 세팅 & 활성화

본 프로젝트는 Python 3.10 기반 전용 가상환경 `nnunet_env`를 사용합니다.

```bash
# 1. 가상환경 활성화
source /home/iulab0/PycharmProjects/nnUNet/nnunet_env/bin/activate

# 2. 가상환경 파이썬 직접 실행 경로 (스크립트/nohup 실행 시 사용)
/home/iulab0/PycharmProjects/nnUNet/nnunet_env/bin/python

# 3. 환경 및 PyTorch CUDA 동작 검증
/home/iulab0/PycharmProjects/nnUNet/nnunet_env/bin/python /home/iulab0/PycharmProjects/nnUNet/agent/test_dummy_nnunet.py
```

---

## 📂 3. 데이터셋 및 결과 디렉토리 구조

```
/home/iulab0/PycharmProjects/nnUNet/
├── Openedsdata2019/
│   ├── Semantic_Segmentation_Dataset/    # OpenEDS 2019 훈련/검증 데이터 (PNG + NPY)
│   ├── Sequence_Extracted/               # 시퀀스 데이터 (Subject별 연속 프레임)
│   └── Sequence_PseudoLabels_400/        # 400x400 모델로 생성된 시퀀스 의사라벨 (.npy)
├── nnUNet_results/
│   ├── VanillaUNet_400x400/              # Phase 1: 400x400 PlainConvUNet 체크포인트 (checkpoint_best.pth)
│   └── TemporalUNet_400x400/             # Phase 3: ConvLSTM 템포럴 디코더 체크포인트
└── agent/                                # 주요 파이프라인 실행 스크립트 모음
```

---

## 🚀 4. 학습 및 파이프라인 구동 가이드

### 🌟 [옵션 A] 전체 파이프라인 1-클릭 마스터 자동 구동 (권장)
마스터 체이닝 스크립트([run_pipeline_400.py](file:///home/iulab0/PycharmProjects/nnUNet/agent/run_pipeline_400.py)) 하나로 **Phase 1 학습 $\rightarrow$ Phase 2 의사라벨 생성 $\rightarrow$ Phase 3 템포럴 학습**까지 연쇄적으로 자동 실행됩니다.

```bash
source /home/iulab0/PycharmProjects/nnUNet/nnunet_env/bin/activate
cd /home/iulab0/PycharmProjects/nnUNet

# 백그라운드 마스터 실행 (IDE/터미널 종료 후에도 안전하게 유지)
nohup python -u agent/run_pipeline_400.py > run_pipeline_400.log 2>&1 &

# 실행 상태 및 로그 실시간 모니터링
tail -f run_pipeline_400.log
```

---

### 🔨 [옵션 B] 단계별 수동 구동 방법

#### Phase 1: 400×400 Vanilla nnUNet (`PlainConvUNet`) 학습
```bash
nohup python agent/train_vanilla_400.py > train_vanilla_400.log 2>&1 &
```
* **학습 전략**:
  * `[120:520, 0:400]` 메모리 크롭으로 $400 \times 400$ 셰이프 추출
  * `min_epochs = 100` (최소 100에폭 학습 보장)
  * `ReduceLROnPlateau` (손실 정체 시 LR을 $1/2$씩 줄여 Sub-pixel 경계 정밀화)
  * Early Stopping (`patience = 60`, `max_epochs = 2000` 무제한 상한)
* **결과 저장**: `nnUNet_results/VanillaUNet_400x400/checkpoint_best.pth`

#### Phase 2: 400×400 시퀀스 의사라벨(Pseudo-Labels) 생성
```bash
python agent/generate_pseudo_labels_400.py
```
* **결과 저장**: `Openedsdata2019/Sequence_PseudoLabels_400/` (전체 시퀀스 프레임 400×400 `.npy` 마스크 저장)

#### Phase 3: 400×400 ConvLSTM TemporalUNet 학습
```bash
nohup python agent/train_temporal_400.py > train_temporal_400.log 2>&1 &
```
* **학습 전략**: Phase 1의 Frozen 인코더 가중치 + ConvLSTM Bottleneck 디코더 시퀀스 학습 ($T=3$)
* **결과 저장**: `nnUNet_results/TemporalUNet_400x400/checkpoint_best.pth`

---

## 📊 5. Weights & Biases (WandB) 모니터링

모든 학습 스크립트에는 **WandB 실시간 모니터링**이 내장되어 있습니다.

* **WandB 프로젝트**: `nnUNet_400x400`
* **WandB 대시보드**: [https://wandb.ai/hyun02051-hongik-university/nnUNet_400x400](https://wandb.ai/hyun02051-hongik-university/nnUNet_400x400)
* **추적 지표**: `train_loss`, `val_loss`, `learning_rate`, `epoch_duration_sec`

---

## 📜 6. `agent/` 스크립트별 역할 및 설명

| 스크립트 명 | 역할 설명 |
| :--- | :--- |
| [train_vanilla_400.py](file:///home/iulab0/PycharmProjects/nnUNet/agent/train_vanilla_400.py) | 400×400 온더플라이 크롭 기반 Vanilla nnUNet (`PlainConvUNet`) 학습 스크립트 (WandB 연동) |
| [generate_pseudo_labels_400.py](file:///home/iulab0/PycharmProjects/nnUNet/agent/generate_pseudo_labels_400.py) | Phase 1 최적 가중치로 400×400 시퀀스 의사라벨 생성 스크립트 |
| [sequence_dataset_400.py](file:///home/iulab0/PycharmProjects/nnUNet/agent/sequence_dataset_400.py) | 400×400 시퀀스 프레임 및 의사라벨 PyTorch DataLoader 모듈 |
| [train_temporal_400.py](file:///home/iulab0/PycharmProjects/nnUNet/agent/train_temporal_400.py) | 400×400 Frozen Encoder + ConvLSTM Bottleneck TemporalUNet 학습 스크립트 |
| [run_pipeline_400.py](file:///home/iulab0/PycharmProjects/nnUNet/agent/run_pipeline_400.py) | Phase 1 PID 감시 $\rightarrow$ Phase 2 $\rightarrow$ Phase 3 자동 연쇄 구동 마스터 스크립트 |
| [temporal_unet.py](file:///home/iulab0/PycharmProjects/nnUNet/agent/temporal_unet.py) | Frozen Encoder + TemporalDecoder 통합 PyTorch 모델 정의 |
| [temporal_decoder.py](file:///home/iulab0/PycharmProjects/nnUNet/agent/temporal_decoder.py) | Bottleneck ConvLSTM 기반 Temporal Decoder 네트워크 구현체 |
| [convlstm.py](file:///home/iulab0/PycharmProjects/nnUNet/agent/convlstm.py) | 2D ConvLSTM Cell 및 시퀀스 레이어 모듈 |
| [evaluate_temporal.py](file:///home/iulab0/PycharmProjects/nnUNet/agent/evaluate_temporal.py) | 템포럴 모델 시퀀스 추론 및 Dice 평가 스크립트 |
| [benchmark_latency.py](file:///home/iulab0/PycharmProjects/nnUNet/agent/benchmark_latency.py) | GPU/CPU 실시간 추론 레이턴시(FPS) 측정 스크립트 |

---

## ⚠️ 7. 오랜만에 재방문 시 유의사항 (Gotchas & FAQs)

1. **장시간 프로세스는 무조건 `nohup` 사용**:
   * 실행 시간이 1시간 이상 소요되는 학습/평가 작업은 일반 터미널에서 실행하면 SSH 세션 종료 시 프로세스가 종료됩니다. 반드시 `nohup python ... > log.log 2>&1 &` 형태로 독립 백그라운드로 띄우세요.
2. **GPU OOM 방지 테스트**:
   * 스크립트 수정 후 가볍게 테스트할 때는 GPU 메모리 간섭을 방지하기 위해 `device='cpu'` 및 `max_samples` 수 개로 먼저 동작 테스트를 거치세요.
3. **가상환경 파이썬 사용**:
   * 시스템 기본 파이썬(`python3`)에는 `torch`, `nnunetv2`, `wandb` 패키지가 없을 수 있으므로 항상 `/home/iulab0/PycharmProjects/nnUNet/nnunet_env/bin/python`을 사용하세요.
