# OpenEDS 2019 공식 테스트 셋 (N=1,440장) 최종 성능 평가 보고서

## 1. 개요 (Overview)
`Dataset600_OpenEDS2019` 데이터셋에서 1,000 에폭 단일 세션 학습을 마친 최적 모델(`checkpoint_best.pth`)을 **공식 OpenEDS 2019 테스트 셋 (`Semantic_Segmentation_Test_Dataset`, 총 1,440장)**에 대해 전수 평가한 결과 보고서입니다.

본 모델은 `SphereHead` 및 `DifferentiableCircleRenderer`를 탑재하여 안구 형상을 타원 왜곡 없이 중심점과 반지름 (cx, cy, r)으로 파라미터화된 **수학적 완전 구형(Sphere) 원형 영역**으로 추정합니다.

---

## 2. 공식 테스트 셋 정량적 평가 결과 (N=1,440장 전수)

### 가. 세그멘테이션 성능 (Segmentation Dice)

| 평가 대상 | Dice 스코어 (%) | 설명 |
| :--- | :---: | :--- |
| **공막 (Sclera)** | **93.13%** | 안구 외부 공막 영역 분할 정확도 |
| **홍채 (Iris)** | **95.66%** | 홍채 영역 분할 정확도 |
| **동공 (Pupil)** | **96.33%** | 동공 중심 영역 고정밀 분할 정확도 |
| **전경 평균 (Mean FG Dice)** | **95.04%** | **공식 테스트 셋 3개 클래스 평균 성능** |

### 나. 안구 구형 기하 구조 성능 (Sphere Geometry Metrics)

| 평가 지표 | 측정 수치 | 비고 및 수식 의미 |
| :--- | :---: | :--- |
| **Circle IoU (영역 일치도)** | **72.07%** | 정답 안구 원 영역과의 교집합/합집합 비율 |
| **Center Error (중심 오차)** | **48.75 px** | 안구 중심점 유클리드 거리 오차 |
| **Radius Error (반지름 오차)** | **12.13 px** | 안구 반지름 추정 오차 |
| **Area / (π × r²) (원형 보존비)** | **1.0000** | **수학적 완전 구형 제약 100% 보존 (타원 불가능)** |
| **Sclera Containment Rate** | **96.36%** | 예측 구형 영역 내 공막 픽셀 포함 비율 |

---

## 3. 주요 평가 지표 수식 정의

1. **Soft Dice Score**:
   Dice = (2 × |P ∩ Y|) / (|P| + |Y|)
   (P: 예측 픽셀 마스크, Y: 정답 픽셀 마스크)

2. **Circle IoU**:
   Circle IoU = Area(C_pred ∩ C_gt) / Area(C_pred ∪ C_gt)
   (C_pred: 예측 원 영역, C_gt: 정답 원 영역)

3. **Center Error (중심 오차)**:
   Center Error = √((cx - cx*)² + (cy - cy*)²) [pixels]

4. **Radius Error (반지름 오차)**:
   Radius Error = |r - r*| [pixels]

5. **Sclera Containment Rate (공막 포함률)**:
   Containment Rate = (예측 구형 안구 내부의 공막 픽셀 수) / (전체 공막 픽셀 수)

---

## 4. 공식 테스트 셋 시각화 오버레이 결과

- **1열 (입력 영상)**: 전처리된 입력 영상 (448 × 448)
- **2열 (정답 마스크 + 정답 원)**: 정답 세그멘테이션 및 청색 점선 정답 원
- **3열 (예측 마스크 + 추정 안구 원)**:
  * **예측 클래스**: 파랑(공막, **93.13%**), 초록(홍채, **95.66%**), 빨강(동공, **96.33%**)
  * **노란색 원 (Pred Sphere Circle)**: 추정 원형 안구 (cx, cy, r) (원형 보존비 **1.0000**, 공막 포함률 **96.36%**)

![공식 테스트 셋 시각화 오버레이 결과](./assets/official_test_overlay_results.png)

---

## 5. 요약
1. **공식 테스트 셋 (1,440장) 전수 평가 완료**: Mean FG Dice **95.04%**, 공막 포함률 **96.36%** 달성.
2. **체계적인 보고서 디렉토리화 완료**: `.agents/reports/Official_Test_Report.md` 및 `.agents/reports/assets/official_test_overlay_results.png`에 정돈 저장 완료.
