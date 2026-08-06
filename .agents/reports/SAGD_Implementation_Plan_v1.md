# Vivim+Mamba3 위에 SAGD 적용: 타당성 평가 및 구현 계획서

## 1. 배경 요약 및 문제 정의

### 1.1 현 상황
- **모델**: nnUNet v2 프레임워크 위에 Vivim(Video Vision Mamba) 백본 + TemporalMambaBlock(TMB) 탑재
- **로스**: WeakMed(CVPR 2025) 기반 M2B + ScleraContainment + DiceCE 복합 손실
- **데이터**: OpenEDS 2019 Sequence (400×640, T=3 temporal window)
- **문제**: 시연 서버 디플로이 시 **동공 검출 부정확 + 낮은 confidence → pye3d 3D 안구 모델 피팅 실패 → 시선 추적 스코어 목표치 하회**

### 1.2 문제 원인 재진단
- 기존 가설: pye3d 기하 연산의 낡은 알고리즘 → **기각됨** (RITnet, 2D C++ 모듈은 아이볼 잘 잡힘)
- 새 가설: **Vivim+Mamba3 모델의 동공 세그멘테이션이 육안으론 괜찮아 보이나, sub-pixel 정확도와 contour smoothness 부족 → `cv2.fitEllipse` 결과의 타원 파라미터 불안정 → pye3d confidence 하락**
- Pupil Labs 코드 확인 결과: [detector_2d_plugin.py](file:///home/iulab1/PycharmProjects/nnUNet/_tmp_pupil_ref/pupil_src/shared_modules/pupil_detector_plugins/detector_2d_plugin.py) 에서 confidence가 **1.0으로 하드코딩** — 즉 confidence 연산 자체가 모델에서 오는 게 아니라 **ellipse fit 품질**에 전적으로 의존함

### 1.3 WeakMed 접근법의 한계 (회의 근거)
| 문제 | 상세 |
|:---|:---|
| Box Collapse | SphereHead가 bounding box 크기로 수렴 (원형 형상 학습 실패) — [WeakMEd 문제 보고서](file:///home/iulab1/PycharmProjects/nnUNet/.agents/reports/WeakMEd_Problem_Report.md) |
| Dual-View 미적용 | 원문 $T_1/T_2$ Cross-View 일치성 없이 단일 뷰만 사용 |
| 피처 퓨전 부재 | SphereHead가 저해상도 병목만 참조 → 세밀한 동공 경계 정보 부족 |
| 도메인 편향 | 학습 도메인(OpenEDS 합성 환경)과 시연 도메인(실제 VR HMD) 간 domain gap을 해소하는 메커니즘 없음 |

### 1.4 SAGD 도입 동기
SAGD(Structure-Aware Domain Generalization)는 **Mamba의 order-sensitivity 문제를 구조 인식 직렬화로 해결**하고, **도메인 갭을 스펙트럼 그래프 정렬로 테스트 타임에 보정**하는 프레임워크로, 현재 시스템의 두 가지 핵심 약점을 동시에 공략:
1. **구조 보존 직렬화** → 동공/홍채/공막의 동심원적 구조를 Mamba 시퀀스에 반영 → contour smoothness 향상
2. **도메인 일반화** → 학습 데이터 vs 시연 환경 간 feature drift를 가중치 업데이트 없이 스펙트럼 정렬

---

## 2. SAGD 논문 핵심 요소 및 2D 안구 세그멘테이션 적용 타당성 분석

### 2.1 SAGD 3대 핵심 컴포넌트

| 컴포넌트 | 원문 (3D Point Cloud) | 2D 안구 적용안 | 타당성 |
|:---|:---|:---|:---|
| **SAS** (Structure-Aware Serialization) | CDS: centroid distance로 BFS 순회<br>GCS: 측지선 heat kernel로 curvature 순회 | **CDS-2D**: 안구 중심(동공)으로부터 동심원 방사형 패치 순서<br>**GCS-2D**: 2D 피처맵 위 Laplacian heat diffusion | ✅ 강한 타당성: 안구 구조가 동공→홍채→공막→배경 동심원 계층 구조이므로 centroid-to-boundary 직렬화가 자연스러움 |
| **HDM** (Hierarchical Domain-Aware Modeling) | Intra-domain Mamba$^p$ / Mamba$^q$ 분기 후 interleave → Mamba$^f$ 퓨전 | **ISM**: 소스 도메인(OpenEDS) 내부 구조 안정화<br>**IRF**: 시연 도메인 피처와 interleave 퓨전 | ⚠️ 조건부 타당성: 학습 시 단일 소스 도메인(OpenEDS)만 사용 중. 다중 소스(RITnet pretrained, real HMD captures 등) 또는 **multi-augmentation 기반 pseudo-domain** 전략 필요 |
| **SGA** (Spectral Graph Alignment) | 테스트 타임 GFT → 소스 프로토타입 방향 스펙트럼 이동 → iGFT | **SGA-2D**: 병목 피처맵을 그래프 신호로 취급, 학습 도메인 프로토타입 방향 스펙트럼 정렬 | ✅ 강한 타당성: 시연 서버 추론 시 도메인 갭 보정의 핵심. 가중치 업데이트 없으므로 실시간 가능 |

### 2.2 3D→2D 도메인 전환 시 핵심 차이점

> [!IMPORTANT]
> SAGD 원문은 unordered point cloud ($\mathbb{R}^3$)를 대상으로 설계되었으나, 우리 문제는 **ordered 2D pixel grid** ($\mathbb{R}^2$) 위의 세그멘테이션입니다. 직접 이식이 아닌 **원리 차용 + 구조 재설계**가 필요합니다.

| 차원 | 원문 (3D) | 적용안 (2D) |
|:---|:---|:---|
| 토큰 생성 | FPS + KNN → N patches | 2D 인코더 피처맵의 spatial patch 또는 superpixel |
| 그래프 구성 | 3D 유클리드/측지선 거리 | 2D 피처 공간 거리 + spatial proximity |
| Centroid | 3D point cloud centroid | 동공 중심 좌표 (또는 피처맵 spatial centroid) |
| 측지선 | 3D manifold shortest path | 2D 피처맵 위 diffusion distance |
| Laplacian | 3D graph Laplacian | 2D 피처맵 affinity graph Laplacian |

### 2.3 종합 타당성 판정

> [!TIP]
> **타당성: ✅ 높음 (조건부)**
>
> SAGD의 핵심 원리(구조 인식 직렬화 + 계층적 도메인 모델링 + 스펙트럼 정렬)는 2D 안구 세그멘테이션에 자연스럽게 매핑됩니다.
> 다만 **단일 소스 도메인 문제**와 **2D 그래프 구성 방식**의 재설계가 필수적이며, 이를 아래 구현 계획에 반영합니다.

---

## 3. 아키텍처 설계

### 3.1 전체 파이프라인 (수정 후)

```
입력 X [B, T=3, C=1, H=448, W=448]
          │
  [2D UNet 인코더] (기존 Vivim enc1→enc2→enc3)
          │
  [Bottleneck] → b [B*T, 256, H/8, W/8]
          │
          ▼
  ┌──────────────────────────────────────┐
  │  SAS-2D: Structure-Aware Serialization │
  │                                        │
  │  1. Spatial 피처 토큰화:               │
  │     b를 [B*T, 256, 56, 56] 피처맵에서 │
  │     N개 패치 토큰 추출                 │
  │                                        │
  │  2. CDS-2D:                            │
  │     동공 중심 기반 centroid affinity    │
  │     graph → BFS 순회 → π_CDS          │
  │                                        │
  │  3. GCS-2D:                            │
  │     2D Laplacian heat diffusion 기반   │
  │     curvature descriptor → π_GCS      │
  │                                        │
  │  4. Unified: [fwd_CDS; rev_CDS;       │
  │              fwd_GCS; rev_GCS]         │
  └──────────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────────┐
  │  HDM-2D: Hierarchical Domain Modeling  │
  │                                        │
  │  ISM: Intra-domain Mamba (기존 TMB)   │
  │       소스 도메인 피처 독립 처리       │
  │                                        │
  │  IRF: Interleaved cross-aug fusion     │
  │       augmented view와 original view를 │
  │       interleave → shared Mamba 퓨전   │
  └──────────────────────────────────────┘
          │
          ▼
  [2D UNet 디코더] → seg_logits [B, 4, H, W]
          │
  (테스트 타임에만)
  ┌──────────────────────────────────────┐
  │  SGA-2D: Spectral Graph Alignment     │
  │                                        │
  │  1. 학습 시: 소스 프로토타입 캐싱     │
  │  2. 테스트 시:                         │
  │     - GFT 변환                         │
  │     - 프로토타입 방향 스펙트럼 이동   │
  │     - iGFT 역변환                     │
  │     - 정렬된 피처로 디코딩            │
  └──────────────────────────────────────┘
```

### 3.2 Pseudo-Domain 전략 (단일 소스 도메인 대응)

원문 SAGD는 $K$개 소스 도메인을 전제하나, OpenEDS는 단일 소스입니다. 이를 해결하기 위해 **data augmentation 기반 pseudo-domain 생성**:

| Pseudo Domain | 증강 전략 | 근거 |
|:---|:---|:---|
| $D_1$ (원본) | Identity (no augmentation) | 베이스라인 |
| $D_2$ (밝기 변이) | 감마 보정 0.5~1.5, CLAHE 변형 | IR 카메라 노출 차이 시뮬레이션 |
| $D_3$ (기하 변이) | 랜덤 아핀 + elastic deformation | 안구 위치/크기 변화 시뮬레이션 |
| $D_4$ (노이즈) | Gaussian noise + motion blur | 센서 노이즈 + 안구 운동 blur |

HDM의 ISM에서 원본 뷰를 소스, 증강 뷰를 query로 사용하여 cross-domain 일치성을 학습합니다.

---

## 4. 파일별 변경 계획

### 4.1 새로 생성할 파일

---

#### [NEW] [sadg_serialization.py](file:///home/iulab1/PycharmProjects/nnUNet/models/sadg_serialization.py)

SAGD의 Structure-Aware Serialization (SAS) 2D 적용 모듈.

**핵심 클래스**:
- `CDS2D`: 동공 중심 기반 centroid distance spectrum. 피처 토큰 간 Gaussian affinity graph 구축 → BFS 순회로 동심원 순서 직렬화
- `GCS2D`: 2D graph Laplacian 기반 heat kernel signature 계산 → curvature-aware 직렬화
- `StructureAwareSerializer`: CDS + GCS 양방향(forward/reverse) 통합 시퀀스 생성기

**수식 매핑**:
- Eq. 4: $w_{CDS}(i,j) = \exp(-\|u_i - u_j\|^2 / \sigma^2)$ → 2D 피처 패치 중심 좌표 사용
- Eq. 5–8: GCS heat diffusion + affinity → 2D Laplacian으로 대체
- Eq. 9: $X_{seq} = [X_{\pi_{CDS}}; X_{rev(\pi_{CDS})}; X_{\pi_{GCS}}; X_{rev(\pi_{GCS})}]$

---

#### [NEW] [sadg_hdm.py](file:///home/iulab1/PycharmProjects/nnUNet/models/sadg_hdm.py)

Hierarchical Domain-Aware Modeling 모듈.

**핵심 클래스**:
- `IntraDomainMamba`: 도메인별 독립 Mamba 블록 (Eq. 11)
- `InterDomainFusion`: 구조 정렬 순서에 따른 interleave + shared Mamba (Eq. 12–13)
- `HDM2D`: ISM + IRF 연쇄 파이프라인

---

#### [NEW] [sadg_sga.py](file:///home/iulab1/PycharmProjects/nnUNet/models/sadg_sga.py)

Spectral Graph Alignment 테스트 타임 모듈.

**핵심 클래스**:
- `SpectralGraphAlignment`: Graph Fourier Transform → 프로토타입 정렬 → inverse GFT
- `SourcePrototypeBank`: 학습 중 소스 도메인 프로토타입 EMA 캐싱

**수식 매핑**:
- Eq. 14: $\hat{P}^s_* = (\Phi^t_*)^\top \frac{1}{N_s}\sum X^s_{\pi_*,i}$
- Eq. 15: $\hat{X}^t_{*,i} \leftarrow \alpha_i \hat{X}^t_{*,i} + (1-\alpha_i)(\hat{P}^s_* - \hat{X}^t_{*,i})$

---

#### [NEW] [sadg_loss.py](file:///home/iulab1/PycharmProjects/nnUNet/losses/sadg_loss.py)

WeakMed 로스 대체 SAGD 전용 손실 함수.

**핵심 클래스**:
- `SADGSegLoss`: DiceCE 기반 지도학습 손실 (기존 유지)
- `DomainConsistencyLoss`: pseudo-domain 간 예측 일관성 손실
- `StructuralContrastiveLoss`: CDS/GCS 직렬화 순서가 도메인 불변이 되도록 하는 contrastive 정규화
- `SADGCombinedLoss`: 위 3가지 통합

---

#### [NEW] [sadg_augmentation.py](file:///home/iulab1/PycharmProjects/nnUNet/datasets/sadg_augmentation.py)

Pseudo-domain 증강 파이프라인. 학습 시 배치 내에서 $D_1$~$D_4$ 증강을 적용하여 multi-domain 학습 효과.

---

#### [NEW] [sadg_vivim.yaml](file:///home/iulab1/PycharmProjects/nnUNet/configs/sadg_vivim.yaml)

SAGD 전용 config. 모든 하이퍼파라미터(SAS 파라미터, HDM 설정, SGA 파라미터, 증강 확률, 손실 가중치) 외부화.

---

#### [NEW] [nnUNetTrainer_Vivim_SADG.py](file:///home/iulab1/PycharmProjects/nnUNet/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainer_Vivim_SADG.py)

기존 `nnUNetTrainer_Vivim` 을 상속하여 SADG 파이프라인을 통합하는 트레이너.

---

### 4.2 수정할 파일

---

#### [MODIFY] [vivim_backbone.py](file:///home/iulab1/PycharmProjects/nnUNet/models/vivim_backbone.py)

- **변경**: 병목 피처맵 출력을 SAS 모듈에 전달할 수 있도록 `forward()` 내 hook point 추가
- SphereHead 관련 코드를 ablation 스위치(`cfg.ablation.use_sphere_head`)로 비활성화 가능하게 유지 (이미 구현됨)
- **삭제하지 않음**: 기존 WeakMed 코드는 ablation 스위치로 비활성화만 하고 보존

---

#### [MODIFY] [openeds_dataset.py](file:///home/iulab1/PycharmProjects/nnUNet/datasets/openeds_dataset.py)

- **변경**: `__getitem__`에서 pseudo-domain 증강 적용 옵션 추가
- 증강 파이프라인은 `sadg_augmentation.py`에서 임포트

---

## 5. 구현 단계별 로드맵

### Phase 1: SAS-2D 구현 및 단독 검증 (2~3일)

- [ ] `models/sadg_serialization.py` 구현
  - [ ] `CDS2D`: 2D affinity graph + BFS 직렬화
  - [ ] `GCS2D`: 2D Laplacian heat kernel + curvature 직렬화
  - [ ] `StructureAwareSerializer`: 통합 양방향 시퀀스
- [ ] 단위 테스트: 합성 피처맵에 대해 직렬화 순서의 구조 보존성 검증
- [ ] 시각화: 448×448 피처맵의 CDS/GCS 순서를 시각적으로 확인

### Phase 2: HDM-2D 구현 (1~2일)

- [ ] `models/sadg_hdm.py` 구현
  - [ ] `IntraDomainMamba` (기존 TMB 재활용 가능)
  - [ ] `InterDomainFusion` (interleave + shared Mamba)
  - [ ] `HDM2D` 통합
- [ ] Pseudo-domain 증강 파이프라인 구현 (`datasets/sadg_augmentation.py`)

### Phase 3: SGA-2D 구현 (1~2일)

- [ ] `models/sadg_sga.py` 구현
  - [ ] Graph Fourier Transform / Inverse GFT
  - [ ] 소스 프로토타입 EMA 캐싱
  - [ ] 적응적 정렬 계수 $\alpha_i$
- [ ] 단위 테스트: 합성 도메인 시프트 시나리오에서 정렬 전/후 feature 비교

### Phase 4: 손실 함수 교체 (1일)

- [ ] `losses/sadg_loss.py` 구현
  - [ ] `DomainConsistencyLoss`
  - [ ] `StructuralContrastiveLoss`
  - [ ] `SADGCombinedLoss`
- [ ] `configs/sadg_vivim.yaml` 작성

### Phase 5: 트레이너 통합 및 학습 (2~3일)

- [ ] `nnUNetTrainer_Vivim_SADG.py` 구현
- [ ] 기존 pretrained seg weight 로드 + SADG 모듈 추가 학습
- [ ] nnUNet 1,000 에폭 네이티브 스케줄 준수
- [ ] WandB 대시보드 동기화

### Phase 6: 평가 및 시연 서버 검증 (2~3일)

- [ ] 공식 테스트 셋(1,440장) 정량 평가
  - Foreground Dice (Sclera/Iris/Pupil)
  - **Pupil contour smoothness** (fitEllipse 잔차)
  - **Pupil ellipse parameter stability** (연속 프레임 간 jitter)
- [ ] Pupil Labs 시연 서버 디플로이
  - pye3d confidence 변화 측정
  - Gaze accuracy (visual angle error) 비교

---

## 6. 리스크 및 완화 전략

| 리스크 | 영향 | 완화 |
|:---|:---|:---|
| SAS의 eigendecomposition 연산 비용 | 학습 속도 저하 | 피처맵을 다운샘플(14×14)하여 토큰 수 제한 (N≤196); 학습 시 캐싱 |
| 단일 소스 도메인의 pseudo-domain 한계 | HDM의 cross-domain 학습 효과 미약 | 증강 다양성 확대 + real HMD 소량 미라벨 데이터 추가 |
| 2D→3D 원리 적용 시 정보 손실 | GCS의 측지선 개념이 2D에서 약화 | Heat diffusion은 2D graph에서도 유효; ablation으로 GCS 기여도 검증 |
| nnUNet 스케줄과 SADG 학습 단계 충돌 | 수렴 불안정 | Phase 5에서 warmup 전략 적용: 처음 50에폭 SAS/HDM 동결, 이후 joint fine-tuning |

---

## 7. 검증 계획

### 자동화 테스트

```bash
# Phase 1 단위 테스트
source /home/iulab0/PycharmProjects/nnUNet/nnunet_env/bin/activate
python -m pytest tests/test_sadg_serialization.py -v

# Phase 2~4 통합 테스트
python -m pytest tests/test_sadg_pipeline.py -v

# Phase 5 학습 시작 (nohup)
nohup nnUNetv2_train Dataset600_OpenEDS2019 2d 0 \
  -tr nnUNetTrainer_Vivim_SADG \
  > logs/sadg_train.log 2>&1 &
```

### 수동 검증

- [ ] Pupil Labs 시연 환경에서 A/B 테스트 (기존 U-Mamba vs SADG)
- [ ] pye3d confidence 분포 히스토그램 비교
- [ ] 시선 추적 visual angle error 비교

---

## Open Questions

> [!IMPORTANT]
> **Q1: Pseudo-domain 전략 vs 실제 multi-domain 데이터**
> 
> OpenEDS 이외의 실제 VR HMD 캡처 데이터(라벨 없는 raw 이미지)를 소량이라도 확보 가능한가? HDM의 효과를 극대화하려면 실제 도메인 변이 데이터가 있는 것이 이상적.

> [!IMPORTANT]
> **Q2: SAS 토큰 수 (N)**
> 
> 병목 피처맵 크기가 56×56=3,136인데, eigendecomposition에는 $O(N^3)$ 비용이 듭니다. 토큰 수를 14×14=196으로 줄이면 연산은 가능하나 공간 정보 손실이 있습니다. 적절한 N 값에 대한 의견?

> [!IMPORTANT]
> **Q3: SphereHead 완전 제거 vs 유지**
> 
> WeakMed 로스를 제거하면 SphereHead의 존재 의의도 사라집니다. SphereHead를 완전히 제거하고 순수 4-class 세그멘테이션 + SAGD 도메인 일반화로 가는 것이 맞는 방향인지?

> [!WARNING]
> **Q4: 학습 시간 예측**
> 
> SADG의 SAS(eigendecomposition)와 HDM(3×Mamba 블록)이 추가되면 iteration당 연산량이 기존 대비 약 2~3배 증가할 것으로 예상됩니다. TITAN RTX 기준 1,000 에폭 학습에 약 3~5일 소요 예상. 이 시간 투자가 허용 가능한지?
