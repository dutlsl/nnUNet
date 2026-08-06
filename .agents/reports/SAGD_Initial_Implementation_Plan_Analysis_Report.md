# 📚 초기 SAGD 구현 계획 및 아키텍처 세부 구현 명세 보고서

**작성일**: 2026-08-06  
**대상 프레임워크**: SAGD (Structure-Aware Domain Generalization) + Vivim (Mamba-3 Backbone)  
**소스 코드 브랜치**: `dutlsl/sadg`  

---

## 📌 1. 개요 및 설계 목적 (Executive Summary)

본 보고서는 **안구 세그멘테이션(Eyeball Segmentation: Pupil, Iris, Sclera)**의 타겟 도메인 일반화(Domain Generalization) 능력 극대화를 위해 설계된 **초기 SAGD(Structure-Aware Domain Generalization) 구현 계획** 및 소스 코드 레벨의 **실제 구현 명세 분석 결과**를 정리합니다.

SAGD 아키텍처는 래스터 스캔(Raster-Scan) 방식의 기존 시퀀스 직렬화 한계를 극복하고, 동심원적 안구 구조(Pupil $\rightarrow$ Iris $\rightarrow$ Sclera $\rightarrow$ Background)를 반영하는 **구조 인지형 시퀀스 직렬화(SAS-2D)** 및 **계층적 도메인 모형화(HDM-2D)**, **스펙트럼 그래프 정렬(SGA-2D)**을 결합합니다.

---

## 📌 2. 핵심 3대 모듈 세부 구현 명세

### 1) SAS-2D: 구조 인지형 직렬화 (Structure-Aware Serialization)
- **소스 위치**: [models/sadg_serialization.py](file:///home/iulab1/PycharmProjects/nnUNet/models/sadg_serialization.py)
- **구현 메커니즘**:
  1. **CDS-2D (Centroid Distance Serialization)**:
     - `self.centroid_head`: 바틀넥 피처맵($24\times24$)에 대해 Adaptive Average Pooling + 2층 MLP를 거쳐 동적으로 동공 중심 좌표 $(c_y, c_x)$를 추정.
     - 그리드 좌표와의 Euclidean 거리를 계산하여 정렬(`torch.argsort(dist)`), 동심원 순서(Pupil $\rightarrow$ Iris $\rightarrow$ Sclera $\rightarrow$ Background)의 시퀀스 토큰 정렬을 생성.
  2. **GCS-2D (Graph Cut Serialization)**:
     - 8-연결성(8-connected grid) 공간 인접 마스크(`adj_mask`)와 1x1 Conv 피처 친밀도(Affinity)를 결합하여 Graph Laplacian $L = D - W$ 계산.
     - Laplacian의 **Fiedler vector (2번째로 작은 고유벡터)**를 추출 및 정렬(`torch.argsort(fiedler)`)하여 피처 유사도 경계를 보존하는 전역적 직렬화 정렬 생성.
  3. **StructureAwareSerializer 클래스**:
     - 정방향/역방향 CDS 및 GCS 4개 시퀀스(`fwd_cds`, `rev_cds`, `fwd_gcs`, `rev_gcs`) 및 2D 복원용 역순열(`inv_cds_order`)을 동시 반환.

---

### 2) HDM-2D: 계층적 도메인 모형화 (Hierarchical Domain Modeling)
- **소스 위치**: [models/sadg_hdm.py](file:///home/iulab1/PycharmProjects/nnUNet/models/sadg_hdm.py)
- **구현 메커니즘**:
  1. **ISM (Intra-Domain Sequence Modeling)**:
     - `IntraDomainMamba`: 각 도메인(OpenEDS, Swirski, LPW)의 시퀀스 토큰을 도메인 독립적 독립 Mamba3 블록 + LayerNorm + Residual 구조로 전처리하여 도메인 고유 시퀀스 동역학 수용.
  2. **IRF (Inter-Domain Representation Fusion)**:
     - `InterDomainFusion`: 3개 도메인의 토큰 시퀀스를 `chunk_size=64` 단위로 라운드로빈(Round-Robin) 방식으로 교차 배치(`_interleave`).
     - 공유 Mamba3 블록(`shared_mamba`)을 통과시켜 도메인 간 공통 구조 지식을 상호 교류(Cross-Domain Knowledge Transfer)한 후 원래 도메인 순서대로 재분리(`_deinterleave`).

---

### 3) SGA-2D: 스펙트럼 그래프 정렬 (Spectral Graph Alignment)
- **소스 위치**: [models/sadg_sga.py](file:///home/iulab1/PycharmProjects/nnUNet/models/sadg_sga.py)
- **구현 메커니즘**:
  1. **SourcePrototypeBank**:
     - 주 도메인(OpenEDS 4-Class) 학습 시 클래스당 $K=4$개의 프로토타입 뱅크를 유지하며 EMA (`momentum=0.999`) 업데이트.
     - `inv_cds_order`를 통해 직렬화 토큰을 2D 공간 피처로 복원(Unscramble)한 후 Ground Truth 라벨 공간에 매칭하여 정밀 업데이트.
  2. **SpectralGraphAlignment**:
     - 추론(Inference/Test-time) 및 검증 단계에서 피처 그래디언트 그래프와 소스 프로토타입 뱅크 간 Softmax 확률 정렬을 수행하여 미지의 타겟 도메인 피처를 스펙트럼 공간 상에서 소스 도메인과 부드럽게 정렬.

---

## 📌 3. 멀티 도메인 데이터 로딩 및 손실함수 구현 명세

### 1) 멀티 도메인 데이터 로더 (Multi-Domain DataLoader)
- **소스 위치**: [datasets/multi_domain_dataset.py](file:///home/iulab1/PycharmProjects/nnUNet/datasets/multi_domain_dataset.py)
- **구현 명세**:
  - **Domain 0 (OpenEDS 2019)**: 4-Class 완전 주석 시퀀스 ($T=3$, $192\times192$).
  - **Domain 1 (Swirski)**: Pupil Ellipse 타겟 정보 수용.
  - **Domain 2 (LPW)**: Pupils in the wild 비디오 타겟 정보 수용.
  - 공통 전처리: **RITnet Preprocessor** ($\text{Gamma 0.8} \rightarrow \text{CLAHE 1.5} \rightarrow \text{Normalize } [-1, 1]$).
  - 배치 구성: `collate_multi_domain`을 통해 배치당 도메인별 6개 샘플(총 Batch=18)을 독립 딕셔너리로 그룹화하여 전달.

---

### 2) 손실함수 및 트레이너 통합 (SAGDLoss & Trainer)
- **소스 위치**: [losses/sadg_loss.py](file:///home/iulab1/PycharmProjects/nnUNet/losses/sadg_loss.py), [nnUNetTrainer_Vivim_SADG.py](file:///home/iulab1/PycharmProjects/nnUNet/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainer_Vivim_SADG.py)
- **구현 명세**:
  1. **손실함수 조합**:
     $$L_{total} = w_{seg} \cdot L_{DiceCE} + \text{warmup}(epoch) \cdot \left( w_{dc} \cdot L_{DC} + w_{sc} \cdot L_{SC} \right)$$
  2. **보완 조치 구현 사항**:
     - **배경 쏠림 방지**: `DiceCELoss`에 `class_weights=[0.1, 1.0, 1.0, 1.0]`을 적용하여 배경 픽셀 그래디언트 독점 현상 억제.
     - **보조 도메인 역할 분리**: Swirski/LPW 등 보조 도메인은 마스크 디코딩 없이 토큰 피처 레벨 손실($L_{SC}$, MMD 기반 $L_{DC}$)에만 참여하도록 분리.
     - **웜업 스케줄**: `warmup_epochs=10`을 설정하여 초반 Mamba3 레이어가 안정적으로 마스크 표현력을 학습한 후 도메인 손실이 개입하도록 가용.

---

## 📌 4. 결론 및 향후 검증 계획

구현 계획서 및 소스 코드 전수 분석 결과, **SAS-2D 동심원 직렬화, HDM-2D 계층적 융합, SGA-2D 프로토타입 정렬** 3대 핵심 모듈이 당초 설계 요구사항에 맞춰 무결하게 구현되었음을 검증하였습니다.

보완된 손실함수 가중치 웜업 및 클래스 밸런싱이 적용된 아키텍처로 1000에폭 메인 학습의 정상 수렴을 지속 모니터링하겠습니다.
