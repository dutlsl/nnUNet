# WeakMEd Eyeball 문제 상황 상세 보고서

## 1. 관측 이미지 및 시각화 종합

![WeakMEd Eyeball Overlay Test Results](file:///home/iulab0/.gemini/antigravity-ide/brain/fad4049b-8b80-4d8e-957f-42b0ca5033c2/test_overlay_result.png)

---

## 2. 이미지 샘플별 상세 문제 현상 분석 (Sample #1 ~ #4)

### 2.1 각 열(Column) 구성
- **Column 1 (Input Frame)**: 400×400 크롭 및 448×448 제로 패딩된 안구 입력 프레임
- **Column 2 (Ground Truth Mask)**: 실제 정답 마스크 (배경, 공막, 홍채, 동공)
- **Column 3 (Pred Seg - Vivim)**: Vivim 딥러닝 백본이 추론한 4-class 세그멘테이션 마스크
- **Column 4 (Seg + Eyeball Overlay)**: 세그멘테이션 마스크 위에 **`eyeball_head` 추론 결과**와 **Sclera-derived Bounding Box (시안색 점선 박스)**를 겹쳐 렌더링한 결과

---

### 2.2 핵심 문제 현상 상세 묘사

#### Eyeball 추론 결과의 박스 붕괴 (Box Collapse)
- Column 4에서 별도의 독립된 원형/구형 안구 윤곽선은 **존재하지 않습니다.**
- `eyeball_head`가 추론한 마스크가 안구의 곡선 형태를 전혀 학습하지 못하고, 약한 지도 라벨(Bounding Box) 영역 내부 전체를 1.0으로 채운 **사각형 블록 마스크**로 출력되었습니다.
- 그 결과, `eyeball_head`의 0.5 경계선(Contour)이 시안색 점선 박스의 **외곽 사각 테두리 선과 완전히 일치**하여 사각형 외곽선 형태로만 나타납니다.

#### 샘플별 동일 현상 확인 (Sample #1 ~ #4)
- 4개 샘플 모두 동일하게 2D 동공/홍채/공막 세그멘테이션(Column 3)은 정상적으로 원형/타원 경계선을 추론합니다.
- 그러나 Eyeball Head(Column 4)는 안구의 기하학적 형태(Sphere/Circle)를 탐지하지 못하고, **Bounding Box 테두리 사각형 그대로 칠해버린 사각 마스크**를 출력하고 있습니다.

---

## 3. 정량적 평가 지표 (Validation Set 16,798개 시퀀스 중 800개 검증)

| 평가 항목 | 세그멘테이션 클래스 | Dice Score | 비고 |
|:---|:---:|:---:|:---|
| **Sclera (공막)** | Class 1 | **95.84%** | 2D 세그멘테이션 정상 |
| **Iris (홍채)** | Class 2 | **97.47%** | 2D 세그멘테이션 정상 |
| **Pupil (동공)** | Class 3 | **97.23%** | 2D 세그멘테이션 정상 |
| **Overall Mean Foreground Dice** | **Mean (1~3)** | **96.85%** | 기존 Vivim 백본 성능 유지 |
| **Eyeball Shape Fidelity** | **3D Eyeball Head** | **실패 (Box Collapse)** | **원형 곡선 부재, 사각형 블록 출력** |

---

## 4. 문제 원인 요약

1. **2D Conv 헤드의 자유도**: `eyeball_head`가 픽셀 단위 2D Conv 마스크 `[B, 1, H, W]`로 구성되어 있어 원/구형 형상을 강제하는 수학적 제약이 없었음.
2. **Loss 최소화의 붕괴**: Bounding Box(정사각형 마스크) 손실 함수만 가해지자, 네트워크가 손실을 가장 빨리 낮추기 위해 **Bounding Box 크기 그대로 사각형 마스크를 출력**하는 최적해로 붕괴(Collapse)함.
