# 안구 추론 스코어 저하 원인 규명 및 v3 아키텍처 분석 보고서

> **문서 목적**: v2 모델에서 발생한 안구 영역 추론 스코어 저하 원인을 규명하기 위해, 사용자가 제기한 3가지 핵심 질문(Dual-View $T_1/T_2$ 미적용, 바운딩 박스 $B$의 정체 및 생성이유, 백본-SphereHead 텐서 퓨전 및 경사하강 흐름)을 원문 논문 및 코드 차원에서 정밀 분석하고 v3 설계를 위한 명확한 문제점 보고서를 작성한다.  
> **주의 사항**: 현 단계에서는 어떠한 코드 수정도 진행하지 않으며, 완성도 높은 보고서 작성에만 집중한다.

---

## 📌 1. 사용자 코멘트 세부 분석 및 질의응답 (Detailed Q&A)

### ❓ 질문 1: M2B Loss에서 원문은 $T_1, T_2$ 두 개를 변환받는데 왜 코드에는 $T$ 하나뿐인가?

#### 1) WeakMed 논문 원문의 수식 및 구조 분석
* **원문 논문 (CVPR 2025, Section 3.2)**:
  $$L_{\text{M2B}} = 0.5 \left[ L_{\text{BCE}}(T_1, B) + L_{\text{BCE}}(T_2, B) \right] + 0.5 \left[ L_{\text{Dice}}(T_1, B) + L_{\text{Dice}}(T_2, B) \right]$$
* **논문의 $T_1, T_2$ 존재 이유**:
  WeakMed 논문은 단일 뷰(Single-view) 데이터만 사용하는 것이 아니라, **서로 다른 강한 데이터 증강(Augmentation)이 적용된 2개의 입력 뷰($x_1, x_2$)를 사용하는 샴(Siamese) / Cross-View Consistency 구조**를 기반으로 작성되었습니다.
  - $P_1$: 뷰 1에서 예측된 마스크 $\rightarrow$ M2B 변환 후 $T_1$
  - $P_2$: 뷰 2에서 예측된 마스크 $\rightarrow$ M2B 변환 후 $T_2$
  - 교차 뷰(Cross-view) 변환 마스크 $T_1, T_2$ 각각에 대해 바운딩 박스 지도 $B$를 가함으로써 조명/회전 변형에 강건한 일치성을 유도합니다.

#### 2) 현 v2 코드 구현 상의 한계
* **코드 위치**: `losses/weakmed_loss.py` (`M2BLoss.forward`)
* **현상**: 단일 뷰 입력 텐서(`sphere_mask` $P$) 하나만 신경망을 통과하여 단일 $T$만 생성된 후 손실이 계산됩니다.
* **영향 분석**: 논문에서 핵심으로 제안한 **Dual-view Cross-Augmentation 일치성 손실**이 생략되고 단일 뷰 M2B만 적용되어, 조명이나 안구 시선 변화 시 뷰 간 억제 효과가 떨어집니다.

---

### ❓ 질문 2: $B$가 무엇이며, 사용자가 $B$를 직접 제공하지 않았는데 코드는 무엇을 정답(GT)으로 삼고 있는가?

#### 1) WeakMed 논문에서 $B$의 원래 정의
* **논문의 $B$**: 약한 지도학습(Weakly Supervised Learning) 설정에서 사람이 직접 그린 **바운딩 박스 정답 마스크 (Bounding Box Ground-Truth Mask)**입니다. (박스 내부 $= 1$, 외부 $= 0$)

#### 2) 현 v2 데이터셋 코드에서의 $B$ 자동 생성 방식 (데이터 흐름 추적)
* **코드 위치**: `datasets/openeds_dataset.py` (lines 82~105 및 line 163)
```python
@staticmethod
def _extract_eyeball_bbox(label: np.ndarray) -> np.ndarray:
    # 2D 완전 지도 픽셀 마스크(Sclera=1, Iris=2, Pupil=3)에서 전경 픽셀을 찾음
    eyeball_mask = (label >= 1)
    rows = np.where(eyeball_mask.any(axis=1))[0]
    cols = np.where(eyeball_mask.any(axis=0))[0]
    # 전경 픽셀의 min/max 좌표로 Bounding Box [x1, y1, x2, y2]를 자동 역산 생성!
    y1, y2 = int(rows.min()), int(rows.max()) + 1
    x1, x2 = int(cols.min()), int(cols.max()) + 1
    return np.array([x1_sq, y1_sq, x2_sq, y2_sq], dtype=np.float32)
```
* **정체 규명**: 사용자가 바운딩 박스를 직접 입력하지 않았으나, `OpenEDS400SequenceDataset` 로더가 **정답 2D 세그멘테이션 마스크(`label >= 1`)의 최외곽 좌표(Min/Max Row & Col)를 런타임에 읽어와 의사 바운딩 박스(Pseudo Bounding Box) $B$를 자동 정답으로 합성**하여 손실 함수에 주입하고 있었습니다.

---

### ❓ 질문 3: 경사 하강은 어디서 실행되며, 네트워크 백본과 SphereHead 간 텐서는 어떻게 퓨전되는가?

#### 1) 백본(Vivim)과 SphereHead 간의 텐서 흐름 & 퓨전 메커니즘
* **코드 위치**: `models/vivim_backbone.py` (`VivimBackbone.forward`, lines 102~150)
* **텐서 분기 및 퓨전 메커니즘**:
  ```
  입력 시퀀스 X [B, T=3, C=1, H, W]
            │
      [2D UNet 인코더]
            │
   [Temporal Mamba Bottleneck] ──▶ 병목 피처 맵 b_last [B, 256, H/8, W/8]
            │                                     │
     (분기 A: 디코더)                       (분기 B: SphereHead)
            │                                     │
    UNet Up-sampling & Skip               Global Avg Pool + MLP
            │                                     │
    2D Seg Logits [B, 4, H, W]            (cx, cy, r) [B, 3] ──▶ Circle Renderer
            │                                                      │
            │                                           Sphere Mask [B, 1, H, W]
            └───────────────────────┬──────────────────────────────┘
                                    ▼
                         [WeakMedSphereLoss 퓨전]
  ```
  1. **피처 차원에서의 직접 퓨전 없음**: 백본 디코더의 2D 세그멘테이션 헤드와 SphereHead는 피처 맵을 intermediate 레이어에서 서로 섞지 않고, **공통의 병목 피처 맵 `b_last` $[B, 256, H/8, W/8]$에서 각각 독립 분기**합니다.
  2. **손실 함수 레벨에서의 퓨전 (Coupling)**:
     두 헤드의 출력은 피처 맵이 아닌 **손실 함수(`WeakMedSphereLoss`) 레벨에서 하나로 결합**됩니다.
     $$\text{Total Loss} = \mathcal{L}_{\text{DiceCE}}(\text{Seg Logits}) + \mathcal{L}_{\text{M2B}}(\text{Sphere Mask}) + 0.5 \times \mathcal{L}_{\text{Containment}}(\text{Seg Logits}, \text{Sphere Mask})$$

#### 2) 경사 하강(Gradient Descent) 실행 위치
* **실행 엔진**: `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py` 및 `nnUNetTrainer_Vivim.py` (`train_step` / `run_iteration`)
* **역전파 및 가중치 업데이트 흐름**:
  1. `optimizer.zero_grad()`
  2. `output = model(images)` $\rightarrow$ `seg_logits`, `sphere_mask`, `sphere_params` 계산
  3. `loss = loss_fn(output, target)` $\rightarrow$ 복합 손실 계산
  4. `loss.backward()` $\rightarrow$ **세그멘테이션 디코더, SphereHead, 그리고 공통 백본(Vivim + Mamba)으로 역전파 기울기(Gradient)가 동시 전달**
  5. `optimizer.step()` $\rightarrow$ PyTorch SGD / AdamW 옵티마이저가 전체 네트워크 파라미터를 경사하강법으로 동시에 업데이트

---

## 📌 2. v2 모델에서 안구 스코어 저하가 발생한 근본 원인 (Root Cause Analysis)

위 분석에 기반하여 v2 모델의 평가 스코어가 낮았던 원인을 4가지 구조적 문제로 정리합니다.

1. **Dual-View Consistency 미적용에 따른 표현력 저하**:
   - WeakMed 원문의 2-view 입력 $T_1, T_2$ 샴 네트워크 구조 대신 단일 뷰 $T$만 사용함으로써, 강한 데이터 증강 간 일치성 제약이 부재하여 약한 위생 지도가 불안정하게 작동함.
2. **SphereHead 피처 퓨전 부재 (독립 분기 문제)**:
   - `SphereHead`가 2D 디코더의 고해상도 세그멘테이션 피처를 참조하지 못하고, 오직 $H/8 \times W/8$ 저해상도 병목 피처 `b_last`만 참조하여 $(c_x, c_y, r)$을 회귀하므로 세밀한 공막/안구 경계 위치 정밀도가 떨어짐.
3. **M2B 손실 지도 시 패치 밖 0-Supervision 결함**:
   - $T$와 $B$를 전체 이미지 공간에서 지도해야 박스 밖으로 원이 삐져나가는 현상을 감점하는데, 바운딩 박스 내부만 잘라서 지도하여 구형 영역 크기 추정에 오차가 발생함.
4. **WeakMed 논문의 SGD 스케줄과 nnUNet 스케줄 간 격리 문제**:
   - 논문은 짧은 16 에폭 + 0.01 lr 디케이 스케줄을 사용하나, nnUNet 1,000 에폭 세션에서는 손실 가중치 $\lambda_{\text{m2b}}$와 $\lambda_{\text{contain}}$의 밸런싱이 초기에 균형을 잃고 2D 세그멘테이션 손실에 치우치는 현상이 발생함.

---

## 📌 3. v3 개선 방향 설계 (Roadmap)

다음 버전(v3) 아키텍처 및 손실 함수 설계를 위한 핵심 개선 로드맵입니다:

1. **아키텍처 레포지셔닝 (피처 퓨전 구현)**:
   - 병목 피처 단독 참조 방식 대신, 디코더의 고해상도 피처($H/2 \times W/2$ 및 $H \times W$)와 2D Seg Logits을 `SphereHead` 입력에 Cross-Attention 또는 Concat 퓨전하는 구조 도입.
2. **Dual-View $T_1, T_2$ M2B Loss 정밀 구현**:
   - 훈련 시 입력 시퀀스에 2가지 서로 다른 Spatial Augmentation을 가해 $T_1, T_2$를 생성하고 원문 Eq. 3의 Cross-view M2B 손실 완벽 복원.
3. **바운딩 박스 $B$ 생성 및 전역 지도 정밀화**:
   - 전체 이미지 공간 $[H, W]$ 상에서 $T_{\text{full}}$과 $B_{\text{full}}$ 간 손실 계산을 엄격히 적용하여 안구 경계 이탈 억제.

---

## 📌 4. 요약 및 보고서 저장 위치

* **보고서 아티팩트 문서**: [.agents/reports/v3_architectural_problem_analysis_report.md](file:///home/iulab0/PycharmProjects/nnUNet/.agents/reports/v3_architectural_problem_analysis_report.md)
* **상태**: 분석 완결 (코드 수정 진행하지 않음)
