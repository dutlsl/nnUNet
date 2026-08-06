---
name: research-code-harness
description: Mandatory rules, architectural constraints, directory layout, and checklists for generating AI/CV research code (eyeball-3d-research).
---

# AI/CV 연구용 코드 생성 하네스 (Research Code Generation Harness)

> **목적**: AI 에이전트가 연구용 코드를 생성할 때 반드시 준수해야 할 구조적 규칙과 금지사항을 정의한다.  
> **적용 범위**: `eyeball-3d-research` 프로젝트 및 이후 파생 연구 브랜치 전체  
> **작성 근거**: `feature/temporal-convlstm` 브랜치 실험 과정에서 발생한 시행착오 분석

---

## 📌 Part 1. 이전 실험에서 발생한 시행착오 기록 (Lesson Learned)

1. **하드코딩된 경로·상수의 산재**: 정규화 상수 (`mean=86.45`, `std=39.94`), 크롭 영역 (`[120:520, 0:400]`), 데이터 디렉토리 경로가 여러 파일에 리터럴로 중복 작성됨.
2. **모듈 간 책임 경계 부재**: Loss 클래스, Dataset 클래스가 `train.py` 내부에 인라인으로 정의되어 다른 모듈에서 재사용 불가능.
3. **실험 재현성 부재**: 하이퍼파라미터가 함수 기본 인자로만 설정되고, 시드 고정 및 설정 스냅샷 저장이 누락됨.
4. **환경·의존성 런타임 실패**: `psutil` 등 무단 외부 패키지 import, Deprecated PyTorch API 사용.
5. **학습 스케줄 임의 간섭 및 평가 셋 혼동**: 커스텀 에폭 컷오프/Early Stopping으로 nnUNet 학습 스케줄을 방해하거나, 검증 셋과 공식 테스트 셋(1,440장)을 혼동하여 보고함.

---

## 📌 Part 2. 연구용 코드 설계 원칙 (Mandatory Rules)

### 규칙 1: 코드와 설정(Config)의 완전한 분리
- 학습률, 배치 크기, 손실 가중치($\lambda$), 크롭 좌표, 정규화 상수, 에폭 수, 시드값 등 모든 파라미터는 **YAML 설정 파일** (`configs/*.yaml`)로 외부화한다.
- 코드 내부에 리터럴 값을 하드코딩하는 것을 **엄격히 금지**한다.

### 규칙 2: 데이터 → 모델 → 손실 → 트레이너 단방향 모듈화
- `datasets/`: 데이터 로딩, 전처리, 증강만 담당 (모델/Loss 의존 금지).
- `models/`: Pure PyTorch 텐서 연산 및 `forward()`만 정의 (Loss 연산 섞지 않음).
- `losses/`: 독립된 손실 함수 클래스 모음 (모델 아키텍처 의존 금지).
- `train.py`: 위 모듈들을 Config 기반으로 조립하고 에폭 루프만 실행. 인라인 클래스 정의 금지.

### 규칙 3: 어블레이션 스위치(Ablation Switch) 기반 실험 체계
- 논문 표(Ablation Table)에 들어갈 제안 기법들은 Config의 스위치 플래그 (`cfg.ablation.use_weakmed`, `cfg.ablation.use_causal_ot` 등)로 제어한다.
- 실험 변형마다 별도의 `train_xxx.py` 스크립트를 생성하지 않고, **단일 `train.py` + Config 교체**로 모든 실험을 처리한다.

### 규칙 4: 실험 재현성 보장
- 학습 시작 전 반드시 시드 고정 유틸 (`random`, `np`, `torch`, `cuda`)을 호출한다.
- 학습 개시 시 `output_dir/config_snapshot.yaml` 저장 및 `checkpoint_best.pth`에 전체 Config 메타데이터를 저장한다.
- WandB run name은 Config 이름 + 타임스탬프로 고유 자동 생성한다.

### 규칙 5: 환경 안전성 및 경로 관리
- `psutil` 등 의존성 목록에 없는 외부 패키지를 사전 확인 없이 import하지 않는다.
- `torch.amp.autocast('cuda')` 등 최신 API 표준을 준수하고 Deprecated API 사용을 금지한다.
- 절대 경로를 하드코딩하지 않으며 환경 변수(`DATA_ROOT`) 또는 Config를 통해 주입한다.

### 규칙 6: Native nnUNet Training Schedule 준수 및 Official Test Set 평가 엄수 (핵심)
- **nnUNet 학습 스케줄 간섭 절대 금지**: `nnUNetTrainer` 및 `nnUNetv2_train` 실행 시 1,000 에폭, PolyLR decay, 250 train / 50 val iteration/epoch 스케줄에 임의로 간섭하거나 커스텀 Early Stopping, Hardcoded Epoch cut-off를 추가하는 행위를 **엄격히 금지**한다. 학습은 반드시 nnUNet 본래 네이티브 스케줄러에 위임한다.
- **공식 테스트 셋 평가 필수**: 최종 성능 평가는 반드시 **공식 테스트 셋 (`Semantic_Segmentation_Test_Dataset`, 1,440장)**에 대해 진행하며, 검증 셋(validation set)을 최종 테스트 셋으로 보고하지 않는다.
- **단일 통합 평가 스크립트 유도**: 평가 스크립트는 임시 파편화 스크립트를 생성하지 말고 [`evaluate_test.py`](file:///home/iulab0/PycharmProjects/nnUNet/evaluate_test.py) 단일 파일로 깔끔하게 유지한다.
- **한국어 전용 보고서 및 디렉토리화**: 모든 실험 결과 보고서는 100% 한국어로 작성하고, 리포지토리 루트에 파일들을 방치하지 않으며 `.agents/reports/Official_Test_Report.md` 및 `.agents/reports/assets/`에 정돈 저장한다.

### 규칙 7: WandB 로깅 위생 및 백그라운드 무의미 프로세스/로그 동시 청소 수칙 (필수)
- **실험적 검증 및 디버깅 실행 시 WandB Offline 준수**: 0에폭 동작 확인, 1000에폭 스트레스 테스트, 빠른 코드/손실 함수 디버깅 등 무의미하거나 임시 검증 목적의 백그라운드 프로세스는 `WANDB_MODE=offline`으로 실행하여 WandB 대시보드 오염을 방지한다.
- **무의미 프로세스 강제 종료 시 동시 청소**: 백그라운드 프로세스를 도중에 강제 종료(`kill`/`pkill`)하거나 10에폭 미만(또는 0에폭)에서 중단되는 무의미한 시도가 발생한 경우, **프로세스 종료와 동시에 불필요한 로그 파일(`logs/*.log`) 및 WandB Run을 즉시 삭제**하여 로그 및 대시보드를 항상 깔끔하게 유지한다.
- **WandB Clean Policy (10에폭 미만 미태그 Run 삭제 원칙)**: 메인 실험으로서 태그(Tag)가 지정되어 있거나 10에폭 이상 정상 진행되어 분석 가치가 있는 Run만 WandB에 보존하고, 태그 없는 10에폭 미만 실패/테스트 Run은 정기적으로 청소한다.

---

## 📌 Part 3. 프로젝트 디렉토리 레이아웃 (`eyeball-3d-research`)

```
eyeball-3d-research/
├── configs/                        # YAML 설정 파일 저장소
│   ├── baseline.yaml               # Exp 1: 순수 2D→3D 기하 투영
│   ├── weakmed_sphere.yaml         # Exp 2: + WeakMed Sphere Loss
│   └── proposed_full.yaml          # Exp 3: 최종 제안 모델
│
├── datasets/                       # 데이터 로더 및 전처리
│   └── openeds_dataset.py          # OpenEDS 2D 마스크 로더 (Sequence T=3 지원)
│
├── models/                         # 신경망 아키텍처
│   ├── vivim_backbone.py           # Vivim 백본
│   └── sphere_head.py              # Differentiable Circle/Sphere Renderer Head
│
├── losses/                         # 손실 함수 모듈
│   └── weakmed_loss.py             # WeakMed M2B & Sclera Containment Loss
│
├── utils/                          # 유틸리티 (시드 고정, 수식, 메트릭)
│   ├── seed.py
│   ├── sphere_metrics.py           # Circle IoU, 중심 오차, 반지름 오차 계산
│   └── metrics.py                  # Dice Score 계산
│
├── .agents/reports/                # 표준 보고서 저장소
│   ├── Official_Test_Report.md     # 한국어 전용 공식 테스트 셋 최종 평가 리포트
│   └── assets/                     # 시각화 오버레이 이미지 저장소
│
├── train.py                        # 학습 메인 스크립트
└── evaluate_test.py                # 공식 테스트 셋 단일 평가 스크립트
```

---

## 📌 Part 4. 에이전트 코드 생성 금지 패턴 (Anti-Patterns)

- ❌ nnUNet 학습 스케줄러에 커스텀 Early Stopping이나 Hardcoded Epoch 조기 종료 삽입 금지 → nnUNet 네이티브 1,000 에폭 스케줄 준수
- ❌ 검증 셋(validation set)을 최종 테스트 셋으로 보고하거나 거짓/옛날 로그 인용 금지 → 공식 테스트 셋(1,440장) 전수 평가
- ❌ `train.py` 내부에 `class DiceLoss(nn.Module):` 인라인 선언 금지 → `losses/`로 분리
- ❌ `img[120:520, 0:400]` 크롭 좌표나 `86.45` 정규화값 리터럴 작성 금지 → Config로 분리
- ❌ `train_baseline.py`, `train_weakmed.py` 등 실험별 파편화 파일 생성 금지 → `train.py --config configs/*.yaml` 또는 native `nnUNetv2_train` 활용
- ❌ `/home/iulab0/...` 절대 경로 하드코딩 금지 → `cfg.data.root` 또는 환경변수 주입
- ❌ 임시/실험적 디버깅 실행을 WandB Online으로 전송하여 대시보드를 오염시키는 행위 금지 → `WANDB_MODE=offline` 사용
- ❌ 무의미하게 종료된 프로세스의 더미 WandB Run 및 로그 파일(`logs/*.log`)을 방치하는 행위 금지 → 즉시 동시 정리 수칙 준수
