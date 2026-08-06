# Vivim+Mamba3 위에 SAGD 적용: 구현 계획서 (코멘트 반영 v2)

## 0. 코멘트 반영 사항 요약

| 코멘트 | 반영 |
|:---|:---|
| WeakMed 모듈 완전 제거, SAGD 브랜치 생성 | ✅ `sadg` 브랜치를 `weakmed-v2` 커밋에서 분기 완료 |
| 전처리는 `vivim_mamba_weakmed_spec.md` 기준 | ✅ Gamma 0.8 → CLAHE 1.5 → Normalize [-1,1], 192×192 인풋 |
| SphereHead 완전 제거 | ✅ 순수 4-class 세그멘테이션 + SAGD 도메인 일반화 |
| Swirski / LPW 실제 multi-domain 데이터 활용 | ✅ 3개 소스 도메인(OpenEDS, Swirski, LPW) 구성 |
| Phase 분리 불필요 — 전면 구현 후 ablation | ✅ 단일 구현 Phase + 학습 후 ablation 연구 |
| nnUNetv2 스케줄 충돌 없이 적용 | ✅ epoch당 ~1분 유지, 시간 재추정 반영 |

---

## 1. 브랜치 및 전처리 규격

### 1.1 Git 브랜치
- **소스 브랜치**: `weakmed-v2` (커밋 `5617e67`)
- **SAGD 브랜치**: `sadg` ← 현재 HEAD ✅

### 1.2 전처리 파이프라인 (RITnet 모사, [spec.md](file:///home/iulab1/PycharmProjects/nnUNet/.agents/reports/vivim_mamba_weakmed_spec.md) §4 준수)

```
Raw IR 안구 영상 (임의 해상도)
    │
    ▼ Resize → 192 × 192 (letterbox or center crop)
    │
    ▼ Gamma Correction (γ=0.8)
    │   table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
    │   cv2.LUT(gray, table)
    │
    ▼ CLAHE (clipLimit=1.5, tileGridSize=(8,8))
    │
    ▼ Normalize → [-1.0, 1.0]
    │   transforms.Normalize([0.5], [0.5])
    │
    ▼ 텐서 [B, T=3, C=1, H=192, W=192]
```

### 1.3 병목 피처맵 크기 (SAS 토큰 수 산정)
- 입력 192×192 → 인코더 3-stage 다운샘플 (×8) → **병목 피처맵 24×24**
- SAS 토큰 수 N = 24×24 = **576** (eigendecomposition $O(576^3)$ ≈ 합리적 연산량)

---

## 2. 소스 도메인 구성 (실제 Multi-Domain)

| 도메인 | 데이터셋 | 경로 | 라벨 | 샘플 수 |
|:---|:---|:---|:---|:---|
| $D_1$ (Primary) | OpenEDS 2019 | `/home/iulab1/PycharmProjects/nnUNet/Openedsdata2019/` | 4-class seg mask (bg/sclera/iris/pupil) | ~12,000 |
| $D_2$ (Swirski) | Swirski Dataset | `/home/iulab1/PycharmProjects/transUnet/Swirski_Dataset/` | Pupil ellipse annotation → seg mask 변환 | ~3,764 (4 subsets × 941) |
| $D_3$ (LPW) | Labelled Pupils in the Wild | `/home/iulab1/PycharmProjects/transUnet/LPW/` | Pupil center annotation → seg mask 변환 | 22 sequences |

> [!NOTE]
> Swirski/LPW는 ellipse/center 형태 라벨이므로 pupil-only binary mask로 변환 후 사용합니다. OpenEDS의 4-class 라벨을 primary supervised loss에 사용하고, Swirski/LPW는 HDM의 cross-domain 일치성 학습에 활용합니다.

---

## 3. 아키텍처 설계 (WeakMed 완전 제거 후)

### 3.1 전체 파이프라인

```
입력 X [B, T=3, C=1, H=192, W=192]
          │
  [2D UNet 인코더] (enc1→enc2→enc3)
          │
  [Bottleneck] → b [B*T, 256, 24, 24]
          │
          ▼
  ┌──────────────────────────────────────┐
  │  SAS-2D: Structure-Aware Serialization │
  │                                        │
  │  1. 피처맵 → N=576 패치 토큰           │
  │  2. CDS-2D: 동공 중심 기반 BFS 직렬화 │
  │  3. GCS-2D: Heat diffusion 직렬화     │
  │  4. 통합: [fwd_CDS; rev_CDS;          │
  │           fwd_GCS; rev_GCS]            │
  └──────────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────────┐
  │  HDM-2D: Hierarchical Domain Modeling  │
  │                                        │
  │  ISM: domain별 독립 Mamba 처리         │
  │  IRF: interleave → shared Mamba 퓨전   │
  │  (OpenEDS / Swirski / LPW 3개 도메인) │
  └──────────────────────────────────────┘
          │
          ▼
  [2D UNet 디코더] → seg_logits [B, 4, 192, 192]
          │
  (테스트 타임에만)
  ┌──────────────────────────────────────┐
  │  SGA-2D: Spectral Graph Alignment     │
  │  프로토타입 캐싱 + 스펙트럼 정렬      │
  └──────────────────────────────────────┘
```

### 3.2 삭제 대상 (WeakMed 관련 전량)
- `losses/weakmed_loss.py` → **삭제** (M2B, ScleraContainment, WeakMedSphereLoss 전체)
- `VivimBackbone` 내 `SphereHead` → **삭제**
- `nnUNetTrainer_Vivim` 내 WeakMed loss 호출 → **삭제**

---

## 4. 파일별 변경 계획

### 신규 생성

| 파일 | 내용 |
|:---|:---|
| [NEW] `models/sadg_serialization.py` | SAS-2D: CDS2D + GCS2D + StructureAwareSerializer |
| [NEW] `models/sadg_hdm.py` | HDM-2D: IntraDomainMamba + InterDomainFusion + HDM2D |
| [NEW] `models/sadg_sga.py` | SGA-2D: SpectralGraphAlignment + SourcePrototypeBank |
| [NEW] `losses/sadg_loss.py` | DiceCE + DomainConsistencyLoss + StructuralContrastiveLoss |
| [NEW] `datasets/multi_domain_dataset.py` | OpenEDS + Swirski + LPW 통합 multi-domain 데이터로더 |
| [NEW] `configs/sadg_vivim.yaml` | 전체 하이퍼파라미터 YAML (전처리, SAS, HDM, SGA, loss weights) |
| [NEW] `nnunetv2/.../nnUNetTrainer_Vivim_SADG.py` | SAGD 통합 트레이너 |

### 수정

| 파일 | 변경 |
|:---|:---|
| [MODIFY] `models/vivim_backbone.py` | SphereHead 삭제, SAS→HDM hook point 추가 |
| [DELETE] `losses/weakmed_loss.py` | WeakMed 로스 전체 삭제 |

### 삭제하지 않고 보존 (참조용)
- `nnUNetTrainer_Vivim.py` 원본은 유지 (ablation 비교 baseline)

---

## 5. 구현 로드맵 (단일 Phase 전면 구현)

> [!IMPORTANT]
> 코멘트 반영: Phase를 나누지 않고 **SAS + HDM + SGA + Loss 전체를 한번에 구현하고 학습** 후, 성과를 확인한 뒤 ablation 연구를 진행합니다.

### Phase 1: 전면 구현 + 학습 (5~7일)

- [ ] WeakMed 모듈 삭제 (weakmed_loss.py, SphereHead)
- [ ] `models/sadg_serialization.py` — SAS-2D 전체
- [ ] `models/sadg_hdm.py` — HDM-2D 전체
- [ ] `models/sadg_sga.py` — SGA-2D 전체
- [ ] `losses/sadg_loss.py` — SAGD 통합 손실
- [ ] `datasets/multi_domain_dataset.py` — 3-domain 통합 로더
- [ ] `configs/sadg_vivim.yaml`
- [ ] `nnUNetTrainer_Vivim_SADG.py` — 트레이너 통합
- [ ] `vivim_backbone.py` 수정 — SAS/HDM 연동 hook point
- [ ] 1,000 에폭 학습 (nnUNetv2 네이티브 스케줄, nohup)

### Phase 2: 정량 평가 + 시연 서버 검증 (2~3일)

- [ ] 공식 테스트셋 Dice/Pupil contour 평가
- [ ] Pupil Labs 시연 환경 pye3d confidence 측정
- [ ] Gaze accuracy visual angle error 비교

### Phase 3: Ablation 연구 (성과 확인 후)

- [ ] SAS only / HDM only / SGA only 각각 비활성화
- [ ] pseudo-domain vs real multi-domain (Swirski/LPW) 비교
- [ ] CDS-2D vs GCS-2D 기여도 분리
- [ ] 192×192 vs 448×448 해상도 비교

---

## 6. 학습 시간 재추정

> [!NOTE]
> 기존 경험치: nnUNetv2 + Vivim 기준 **1 에폭 < 1분**

| 항목 | 기존 Vivim | SAGD 추가 시 |
|:---|:---|:---|
| Bottleneck Mamba (TMB) | ~0.3s/iter | 유지 |
| SAS eigendecomposition (N=576) | - | +~0.05s/iter (캐싱 가능) |
| HDM (3×Mamba) | - | +~0.15s/iter |
| SGA (학습 시 프로토타입 EMA만) | - | +~0.01s/iter |
| **총 추정** | ~0.3s/iter | **~0.5s/iter** |

- SAS의 eigendecomposition은 **학습 시 epoch 시작 시 1회 계산 후 캐싱**하므로 iteration 단위 부하 미미
- HDM의 추가 Mamba 블록이 주요 비용이나, 192×192 입력 기준 병목 24×24로 연산량 경미
- **1 에폭 ~1.5분 이내 예상** (기존 대비 ~50% 증가, nnUNetv2 스케줄과 충돌 없음)
- **1,000 에폭 총 학습 시간: ~25시간** (TITAN RTX / RTX 3090 기준)

---

## 7. 검증 계획

### 자동화 테스트

```bash
source /home/iulab1/PycharmProjects/nnUNet/nnunet_env/bin/activate

# 통합 테스트 (SAGD 전 모듈)
python -m pytest tests/test_sadg_full.py -v

# 학습 시작 (nohup, 1시간 이상)
nohup nnUNetv2_train Dataset600_OpenEDS2019 2d 0 \
  -tr nnUNetTrainer_Vivim_SADG \
  > logs/sadg_train.log 2>&1 &
```

### 수동 검증

- [ ] Pupil Labs 시연 환경 A/B 테스트
- [ ] pye3d confidence 분포 비교
- [ ] Visual angle error 비교

---

## Open Questions (해결됨)

| 원문 | 해결 |
|:---|:---|
| Q1: Multi-domain 데이터 | ✅ Swirski + LPW 실제 데이터 3-domain 구성 |
| Q2: SAS 토큰 수 | ✅ 192→24×24=576 토큰 (합리적) |
| Q3: SphereHead 제거 | ✅ 완전 제거 결정 |
| Q4: 학습 시간 | ✅ 1에폭 ~1.5분, 총 ~25시간 (스케줄 충돌 없음) |
