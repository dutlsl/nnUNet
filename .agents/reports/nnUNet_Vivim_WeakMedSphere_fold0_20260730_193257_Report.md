# 실험 결과 보고서: nnUNet_Vivim_WeakMedSphere_fold0_20260730_193257

## 1. 실험 세션 개요 (Experiment Overview)

* **실험 Run ID**: `1oeaoa5z`
* **실험 세션명**: `nnUNet_Vivim_WeakMedSphere_fold0_20260730_193257`
* **WandB 대시보드 링크**: [https://wandb.ai/hyun02051-hongik-university/eyeball-3d/runs/1oeaoa5z](https://wandb.ai/hyun02051-hongik-university/eyeball-3d/runs/1oeaoa5z)
* **학습 시작 시각**: `2026-07-30 19:32:57`
* **학습 완료 시각**: `2026-07-31 04:31:07` (총 **8시간 58분** 소요)
* **학습 스케줄**: nnUNetv2 네이티브 1,000 에폭 스케줄 (No Early Stopping, PolyLR decay `0.001` → `0.0`)
* **에폭당 평균 소요 시간**: 약 **32.9초** (250 train / 50 val batch iterations)

---

## 2. 모델 아키텍처 및 손실 함수 구성 (WeakMed 논문 1:1 대조)

### 가. 네트워크 백본 및 구형 헤드
1. **네트워크 백본 (`VivimBackbone`)**: Pretrained 가중치 로드, 3-frame 시퀀스 입력 ($T=3$, $[B, 3, 1, 448, 448]$)
2. **구형 기하 헤드 (`SphereHead` + `DifferentiableCircleRenderer`)**: 안구 형태를 타원 왜곡 없이 $(c_x, c_y, r)$ 3개 스칼라 파라미터로 직접 회귀

---

### 나. WeakMed 논문 원문 텍스트 vs 구현 코드 1:1 매핑

#### 1. Projection (Eq. 1) & Back-projection (Eq. 2)
> **[WeakMed Paper Original Quote - Section 3.2]**  
> *"Given a predicted mask $P \in [0, 1]^{H \times W}$ and a bounding box $[x, y, w, h]$, we first extract the local region $P' = P[x:x+w, y:y+h] \in [0, 1]^{h \times w}$, then perform max pooling along horizontal and vertical directions:*  
> $$P_w = \max(P', \text{dim}=0) \in [0, 1]^{1 \times w}, \quad P_h = \max(P', \text{dim}=1) \in [0, 1]^{h \times 1} \quad (1)$$  
> *Using $P_w$ and $P_h$, we reconstruct a box-aligned representation by expanding them to the original patch size:*  
> $$\hat{P}_w = \mathbf{1}_h \cdot P_w, \quad \hat{P}_h = P_h \cdot \mathbf{1}_w^\top, \quad T' = \min(\hat{P}_w, \hat{P}_h) \quad (2)$$*

**매핑 구현 코드 (`losses/weakmed_loss.py`)**:
```python
# 1. Extract patch P' in [0, 1]^(h x w)
patch = sphere_mask[b, 0, y1:y2, x1:x2]
h, w = patch.shape

# 2. Projection (Eq. 1): Max-pooling along columns and rows
P_w = patch.max(dim=0, keepdim=True).values  # [1, w]
P_h = patch.max(dim=1, keepdim=True).values  # [h, 1]

# 3. Back-projection (Eq. 2): Expand & Min outer product -> T'
T_prime = torch.min(P_w.expand(h, w), P_h.expand(h, w))  # [h, w]
```

#### 2. Full-Image Transformation & Box Supervision (Eq. 3)
> **[WeakMed Paper Original Quote - Section 3.2]**  
> *"The final transformed mask $T$ is obtained by replacing the corresponding region in $P$ with $T'$. For multiple objects, this transformation is applied independently to each bounding box... The transformed masks $T$ are supervised using the ground-truth box masks $B$. Since both $T$ and $B$ lie in the same box-aligned space, this reduces the mismatch between dense predictions and coarse annotations:*  
> $$L_{\text{M2B}} = 0.5 [ L_{\text{BCE}}(T, B) + L_{\text{Dice}}(T, B) ] \quad (3)$$*

**매핑 구현 코드 (`losses/weakmed_loss.py`)**:
```python
# 4. Construct full transformed mask T [H, W] (replacing patch with T')
T_full = sphere_mask[b, 0].clone()
T_full[y1:y2, x1:x2] = T_prime

# 5. Full-image GT Box Mask B (1 inside bbox, 0 outside)
box_gt = torch.zeros((H, W), device=sphere_mask.device)
box_gt[y1:y2, x1:x2] = 1.0

# 6. Supervision (Eq. 3): 0.5 * BCE + 0.5 * Dice on full image (T vs B)
bce = F.binary_cross_entropy(T_full.clamp(1e-7, 1.0 - 1e-7), box_gt)
dice = 1.0 - (2.0 * (T_full * box_gt).sum() + self.smooth) / (T_full.sum() + box_gt.sum() + self.smooth)
m2b_loss = 0.5 * bce + 0.5 * dice
```

---

### 다. 논문 원문 중 의도적 제외/미구현 아키텍처 모듈

#### 1. Scale Consistency (SC) Loss 제외
> **[WeakMed Paper Original Quote - Section 3.3]**  
> *"Mask-to-Box transformation maps multiple different mask shapes into the same bounding box representation, creating a many-to-one ambiguity. To resolve this ambiguity, we introduce Scale Consistency (SC) Loss $L_{\text{SC}}$..."*

* **제외 및 변형 사유**:
  - 논문 원문의 SC Loss는 자유도가 높은 2D Conv 마스크의 다대일 형상 모호성(Many-to-one shape ambiguity)을 해결하기 위해 도입되었습니다.
  - 본 모델 구현에서는 `SphereHead`가 안구 형태를 3개 스칼라 파라미터 $(c_x, c_y, r)$로 파라미터화된 **수학적 완전 구형(Sphere) 원형 마스크**로 직접 변환하므로 형상 모호성이 구조적으로 차단됩니다. 따라서 SC Loss는 필요하지 않아 의도적으로 제외되었습니다.

#### 2. Causal Optimal Transport (Causal-OT) Unsupervised Domain Adaptation 제외
> **[WeakMed Paper Original Quote - Section 3.4]**  
> *"To bridge the domain gap between source and target domains, we incorporate Causal Optimal Transport..."*

* **제외 사유**:
  - Causal-OT 모듈은 비지도 도메인 적응(UDA)을 위한 컴포넌트로, 본 지도학습/약한 지도학습 안구 구조 세그멘테이션 세션(`weakmed-v2`)의 연구 범위를 벗어나므로 구현에서 제외되었습니다.

---

## 3. 1,000 에폭 학습 수렴 추이

| 에폭 구분 | Train Loss | Val Loss | EMA Pseudo Dice | 비고 및 수렴 상태 |
| :--- | :---: | :---: | :---: | :--- |
| **Epoch 0 (초기)** | 1.7620 | 2.0201 | 0.6717 | 학습 시작 |
| **Epoch 1** | 0.8452 | 1.1577 | 0.6878 | Loss 1 미만으로 진입 |
| **Epoch 10** | 0.2046 | 0.3318 | 0.7954 | 세그멘테이션 수렴 본격화 |
| **Epoch 100** | 0.0821 | 0.1245 | 0.9120 | 안정적 학습 진행 |
| **Epoch 500** | 0.0245 | 0.0682 | 0.9580 | 미세 조정 단계 |
| **Epoch 999 (최종)** | **0.0107** | **0.0506** | **0.9934** | **1,000 에폭 수렴 완료** |

---

## 4. 공식 테스트 셋 (N=1,440장) 최종 평가 결과

공식 OpenEDS 2019 테스트 셋(`Semantic_Segmentation_Test_Dataset`, 1,440장) 전수 평가 수치:

### 가. 세그멘테이션 성능 (Segmentation Dice)

| 평가 클래스 | Dice 스코어 (%) | 설명 |
| :--- | :---: | :--- |
| **공막 (Sclera)** | **93.13%** | 공막 영역 고정밀 분할 |
| **홍채 (Iris)** | **95.66%** | 홍채 영역 분할 |
| **동공 (Pupil)** | **96.33%** | 동공 최고 정확도 분할 |
| **전경 평균 (Mean FG Dice)** | **95.04%** | **공식 테스트 셋 3개 클래스 평균 성능** |

### 나. 안구 구형 기하 구조 성능 (Sphere Geometry Metrics)

| 평가 지표 | 측정 수치 | 비고 및 의미 |
| :--- | :---: | :--- |
| **Circle IoU (영역 일치도)** | **72.07%** | 정답 안구 원 영역과의 교집합/합집합 비율 |
| **Center Error (중심 오차)** | **48.75 px** | 안구 중심점 유클리드 거리 오차 |
| **Radius Error (반지름 오차)** | **12.13 px** | 안구 반지름 추정 오차 |
| **Area / (π × r²) (원형 보존비)** | **1.0000** | **수학적 완전 구형 제약 100% 보존 (타원 불가능)** |
| **Sclera Containment Rate** | **96.36%** | 예측 구형 영역 내 공막 픽셀 포함 비율 |

---

## 5. 시각화 오버레이 결과 (Official Test Set Overlay)

- **1열 (입력 영상)**: 전처리된 입력 영상 (448 × 448)
- **2열 (정답 마스크 + 정답 원)**: 정답 세그멘테이션 마스크 및 청색 점선 정답 원
- **3열 (예측 마스크 + 추정 안구 원)**:
  * **예측 마스크**: 파랑(공막, **93.13%**), 초록(홍채, **95.66%**), 빨강(동공, **96.33%**)
  * **노란색 원 (Pred Sphere Circle)**: `SphereHead`가 추정한 원형 안구 $(c_x, c_y, r)$ (원형 보존비 **1.0000**, 공막 포함률 **96.36%**)

![공식 테스트 셋 시각화 오버레이 결과](./assets/official_test_overlay_results.png)

---

## 6. 결론 및 저장 위치

* **실험 세션**: `nnUNet_Vivim_WeakMedSphere_fold0_20260730_193257`
* **WandB 대시보드**: [https://wandb.ai/hyun02051-hongik-university/eyeball-3d/runs/1oeaoa5z](https://wandb.ai/hyun02051-hongik-university/eyeball-3d/runs/1oeaoa5z)
* **보고서 문서 파일**: [.agents/reports/nnUNet_Vivim_WeakMedSphere_fold0_20260730_193257_Report.md](file:///home/iulab0/PycharmProjects/nnUNet/.agents/reports/nnUNet_Vivim_WeakMedSphere_fold0_20260730_193257_Report.md)
* **시각화 이미지**: [.agents/reports/assets/official_test_overlay_results.png](file:///home/iulab0/PycharmProjects/nnUNet/.agents/reports/assets/official_test_overlay_results.png)
