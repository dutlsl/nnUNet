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

---

## 📌 Part 3. 프로젝트 디렉토리 레이아웃 (`eyeball-3d-research`)

```
eyeball-3d-research/
├── configs/                        # YAML 설정 파일 저장소
│   ├── baseline.yaml               # Exp 1: 순수 2D→3D 기하 투영
│   ├── weakmed.yaml                # Exp 2: + WeakMed Loss
│   ├── causal_ot.yaml              # Exp 3: + Causal-OT 적응
│   └── proposed_full.yaml          # Exp 4: 최종 제안 (WeakMed + Causal-OT)
│
├── datasets/                       # 데이터 로더 및 전처리
│   ├── openeds_dataset.py          # OpenEDS 2D 마스크 로더 (4-fold 지원)
│   └── pupil_labs_dataset.py       # Target 도메인 Pupil Labs 이미지 로더
│
├── models/                         # 신경망 아키텍처
│   ├── backbone_2d.py              # 2D 세그멘테이션 백본 (UNet / PVTv2 등)
│   ├── eyeball_3d_regressor.py     # 3D 안구 파라미터 회귀 헤드
│   └── projection_layer.py         # 미분 가능한 3D 구체 → 2D 마스크 투영
│
├── losses/                         # 손실 함수 모듈
│   ├── projection_loss.py          # 기본 2D-3D 일치성 손실
│   ├── weakmed_loss.py             # WeakMed M2B & Scale Consistency
│   └── causal_ot_loss.py           # Granger 인과 그래프 + Sinkhorn OT
│
├── utils/                          # 유틸리티 (시드 고정, 엔트로피, 메트릭)
│   ├── seed.py
│   ├── uncertainty.py
│   └── metrics.py
│
├── train.py                        # 학습 메인 실행 스크립트 (유일한 엔트리포인트)
├── evaluate.py                     # 평가 및 Stuck 발생률 검증
└── main_pipeline.py                # 인퍼런스 및 실시간 상태 버퍼(Prior) 테스트
```

---

## 📌 Part 4. 에이전트 코드 생성 금지 패턴 (Anti-Patterns)

- ❌ `train.py` 내부에 `class DiceLoss(nn.Module):` 인라인 선언 금지 → `losses/`로 분리
- ❌ `img[120:520, 0:400]` 크롭 좌표나 `86.45` 정규화값 리터럴 작성 금지 → Config로 분리
- ❌ `train_baseline.py`, `train_weakmed.py` 등 실험별 파편화 파일 생성 금지 → `train.py --config configs/*.yaml` 활용
- ❌ `/home/iulab0/...` 절대 경로 하드코딩 금지 → `cfg.data.root` 활용
