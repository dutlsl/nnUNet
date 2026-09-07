# Pupil Labs 딥러닝 동공 디텍터 사용자 가이드 및 실행 매뉴얼 (User Guide & Harness Manual)



본 문서는 **Pupil Labs 딥러닝 동공 디텍터 모듈(`Detector2DPlugin`)의 실행 환경, GUI 조작법, 오프라인 검증, 트러블슈팅 및 캘리브레이션/검증 타겟 좌표 설정법을 정리한 유저 매뉴얼**입니다.

---

## 1. 빠른 실행 요약 (Quick Reference)

| 작업 목적 | 실행 명령어 / 조작 방법 | 비고 |
|:---|:---|:---|
| **가상환경 활성화** | `conda activate pupil-umamba` | Python 3.10 + PyTorch CUDA 12 환경 (필수) |
| **Pupil Capture GUI 실행** | `cd ~/PycharmProjects/pupil/pupil_src && python main.py` | 실시간 AR 글래스 수신 & **`pupil_capture.log` 자동 저장** |
| **자동 휘발성 로그 파일** | `cat ~/PycharmProjects/pupil/pupil_capture.log` | `python main.py` 실행 시마다 기존 로그 자동 덮어쓰기 (`mode="w"`) |
| **오프라인 하네스 검증 실행** | `cd ~/PycharmProjects/pupil && python tests/test_validation.py` | 캘리브레이션/밸리데이션 파이프라인 및 Mamba3 매핑 검증 |
| **5-Stack 시연 검증** | `cd ~/PycharmProjects/pupil && python tests/test_5stack_demo.py` | 5회 밸리데이션 스택 누적 및 통계(평균/표준편차) 콘솔 출력 검증 |
| **디텍터 모델 전환 (UI)** | `Eye -> 2D Detector -> Active Model` 드롭다운 | `Mamba3 (T=7)` (신규/기본) / `2D C++` / `RITnet` |
| **Left Eye (Eye 0) 반전** | `Eye -> 2D Detector -> Flip Vertically` 체크 | 광학 거울 180도 뒤집힘 보정 |

---

## 2. 필수 실행 환경 (Environment Setup)

### 2.1 Conda 가상환경 활성화
본 시스템은 **`pupil-umamba`** Conda 가상환경(Python 3.10, PyTorch CUDA 12, NumPy 1.26.4)에서 구동됩니다.

```bash
# 1. Conda 가상환경 활성화
conda activate pupil-umamba

# 2. 필수 환경 변수 확인 (자동 설정)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### 2.2 디렉토리 및 모델 경로 구조

- **Pupil 소스 루트**: `~/PycharmProjects/pupil/pupil_src`
- **Mamba3 (T=7) 모델 및 가중치**:
  - 모델 코드: `pupil_src/shared_modules/pupil_detector_plugins/vivim/` (내장 Vivim-Mamba3 SSM)
  - 사전학습 가중치: `pupil_src/shared_modules/pupil_detector_plugins/best_checkpoint_t7.pth`
- **RITnet 모델 및 가중치**:
  - 모델 코드: `pupil_src/shared_modules/pupil_detector_plugins/densenet.py` (DenseNet2D)
  - 사전학습 가중치: `pupil_src/best_model.pkl`
- **2D C++ 기본 검출기**:
  - C++ 바이너리 라이브러리: `pupil_detectors`

---

## 3. 프로그램 실행 방법 (Execution Guide)

### 3.1 실제 AR 글래스 착용 및 수신 시 (Pupil Capture GUI)

AR 글래스를 시연 컴퓨터 USB 포트에 연결한 후 다음 명령어로 실시간 GUI를 실행합니다:

```bash
cd ~/PycharmProjects/pupil/pupil_src
python main.py
```

- 실행 시 **Pupil Capture 메인 창**과 **안구 카메라 패널(Eye 0 / Eye 1)**이 표시됩니다.

### 3.2 하드웨어 미연결 시 사전 검증 (오프라인 더미 테스트)

AR 글래스 없이 디텍터 연산, 모델 로딩, 80~200+ FPS 추론 지연시간을 사전 검증하려면 하네스 스크립트를 실행합니다:

```bash
cd ~/PycharmProjects/pupil
python tests/test_dummy_harness.py
```

---

## 4. UI 인터페이스 조작 및 설정 가이드 (UI User Manual)

Pupil Capture GUI 좌측/상단 메뉴를 통한 디텍터 설정 방법입니다.

### 4.1 디텍터 선택 및 Active Model 스위칭
1. Eye 창(안구 카메라 창)의 메뉴 아이콘 클릭
2. **`Pupil Detector 2D`** 설정 메뉴 이동
3. **`Active Model`** 드롭다운 항목 선택:
   - **`Mamba3 (T=7)` (신규/기본)**: Video Vision Mamba selective scan 디코더 기반. 7프레임 시계열 컨텍스트 반영, 0.447° 최고 수준 Peak 정확도 (~83+ FPS).
   - **`2D C++`**: Pupil Labs 기존 C++ 기하학 디텍터. <1ms 초고속 추론 및 극소 지터.
   - **`RITnet`**: DenseNet2D 기반 레거시 베이스라인. 안정적인 평균 오차 달성.
4. **`Enable Calibration`** 토글 스위치:
   - `ON`: 캘리브레이션에서 피팅된 LinearRegression 모델로 시선 매핑.
   - `OFF`: 원시 동공 정규화 좌표(`raw pupil norm_pos`)를 시선으로 직접 패스스루.

### 4.2 Eye 0 (왼쪽 안구) 방향 보정
Pupil Core 스마트 글래스의 광학 거울 반사 구조상 Eye 0 카메라는 180도 뒤집힌 상태로 수신됩니다:
- **`Flip Vertically (Eye 0)`**: 체크박스를 켜면 모델 추론 전 상하 반전 후 바르게 세워 추론하며, 결과를 원래 좌표로 원복합니다.
- **`Flip Horizontally (Eye 0)`**: 좌우 반전 보정이 필요한 경우 체크합니다.

### 4.3 반눈(Half-blink), 점프 거부 및 이동평균(EMA) 필터링
- **블링크 및 왜곡 자동 차단**: 눈꺼풀에 동공이 가려지거나 찌그러진 경우(`aspect_ratio < 0.20` 또는 `area < 15.0`) 자동으로 빈 데이터를 반환하여 오류 유입 차단.
- **점프 거부 (Jump Rejection, 40px)**: 1프레임 사이에 동공 중심이 40px 이상 튈 때 4프레임 이하 동안은 노이즈로 간주(`confidence = 0.0`), 5프레임 이상 지속 시 실제 Saccade로 인정.
- **시계열 이동평균(EMA)**: `alpha=0.4` 수준의 지수이동평균을 적용하여 동공 중심점 미세 떨림 60% 흡수.

---

## 5. 입력 규격 및 처리 파이프라인 (Data Pipeline)

| 항목 | Pupil Core 카메라 (실제 입력) | 모델 추론 | 처리 방식 |
|:---|:---|:---|:---|
| **해상도 및 규격** | $192 \times 192$ (정사각형) | 정사각형 입력 규격 | **Native Direct Input**: 인위적인 패딩/슬라이싱 없이 원본 정사각형 프레임을 정규화 후 모델에 직접 입력 및 1:1 역변환 |
| **밝기 정규화** | 동적 IR 조명 변화 | Z-Score Normalization | **Dynamic Normalization**: 프레임별 `(frame - mean) / std` 적용 |

---

## 6. 트러블슈팅 및 문제 해결 (Troubleshooting)

1. **`UnpicklingError` 발생 시**:
   - `main.py` 최상단에 `weights_only=False` 몽키패치가 적용되어 있는지 확인.
2. **`ModuleNotFoundError: No module named 'mamba_ssm'`**:
   - `conda activate pupil-umamba` 가상환경이 켜져 있는지 확인.
3. **카메라 입력이 찌그러지거나 동공이 튀는 현상**:
   - UI에서 `Flip Vertically`가 올바르게 켜져 있는지, `Active Model`이 `Mamba3 (T=7)`로 설정되어 있는지 확인.

---

## 7. 🎯 캘리브레이션 및 검증(Test Validation) 타겟 좌표 설정 매뉴얼

사용자 시선 교정(Calibration) 및 검증(Validation/Accuracy Test) 시 표시되는 타겟 포인트 좌표는 코드 상에 아래 위치에 정의되어 있습니다.

### 7.1 화면 마커 캘리브레이션 및 검증 타겟 좌표 (Screen Marker)
- **파일 경로**: [pupil_src/shared_modules/calibration_choreography/screen_marker_plugin.py](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/calibration_choreography/screen_marker_plugin.py#L71-L119)
- **클래스 및 메서드**: `ScreenMarkerChoreographyPlugin.get_list_of_markers_to_show(self, mode)` (Lines 71~119)

```python
# 캘리브레이션 패턴 정의 (5-Point, 9-Point Grid, 12-Point Dense)
CALIBRATION_PATTERNS = {
    "5-Point (Pupil Labs Default)": [
        (0.5, 0.5), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0),
    ],
    "9-Point (3x3 Grid / Ours)": [
        (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),  # Top row (1, 2, 3)
        (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),  # Middle row (4, 5, 6)
        (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),  # Bottom row (7, 8, 9)
    ],
    "12-Point (4x3 Dense Grid / New)": [
        (0.0, 1.0), (0.333, 1.0), (0.667, 1.0), (1.0, 1.0),
        (0.0, 0.5), (0.333, 0.5), (0.667, 0.5), (1.0, 0.5),
        (0.0, 0.0), (0.333, 0.0), (0.667, 0.0), (1.0, 0.0),
    ],
}

# 밸리데이션 패턴 정의 (Diamond Inward Cross, 4 Corners)
VALIDATION_PATTERNS = {
    "Diamond (Inward Cross / Default)": [
        (0.5, 0.8), (0.8, 0.5), (0.5, 0.2), (0.2, 0.5),
    ],
    "4 Corners (Extreme Boundaries)": [
        (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0),
    ],
}

def get_list_of_markers_to_show(self, mode: ChoreographyMode) -> list:
    if ChoreographyMode.CALIBRATION == mode:
        pattern = getattr(self, "calibration_pattern", "12-Point (4x3 Dense Grid / New)")
        return list(self.CALIBRATION_PATTERNS.get(pattern, self.CALIBRATION_PATTERNS["12-Point (4x3 Dense Grid / New)"]))
    if ChoreographyMode.VALIDATION == mode:
        pattern = getattr(self, "validation_pattern", "Diamond (Inward Cross / Default)")
        return list(self.VALIDATION_PATTERNS.get(pattern, self.VALIDATION_PATTERNS["Diamond (Inward Cross / Default)"]))
    raise ValueError(f"Unknown mode {mode}")
```
- **수정 가이드**:
  - UI 상에서 드롭다운으로 원하는 패턴을 실시간 선택하거나, `CALIBRATION_PATTERNS` / `VALIDATION_PATTERNS` 딕셔너리에 새로운 패턴 좌표 리스트를 추가할 수 있습니다.

### 7.2 단일 마커 캘리브레이션 고정 좌표 (Single Marker)
- **파일 경로**: [pupil_src/shared_modules/calibration_choreography/single_marker_plugin.py](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/calibration_choreography/single_marker_plugin.py#L100)
- **상수 위치**: `SingleMarkerChoreographyPlugin._FIXED_MARKER_POSITION` (Line 100)
```python
_FIXED_MARKER_POSITION = (0.5, 0.5)  # 화면 중앙 고정 마커 좌표
```

### 7.3 오프라인 테스트 하네스 안구 타겟 좌표 (Synthetic Dummy Frame)
- **파일 경로**: [tests/test_dummy_harness.py](file:///home/byeongjun/PycharmProjects/pupil/tests/test_dummy_harness.py#L32-L35)
- **함수 위치**: `make_synthetic_eye(height, width)` (Lines 32~35)
```python
cx, cy = width // 2, height // 2  # 가상 생성 동공 타겟 좌표
```

### 7.4 Accuracy Visualizer Outlier Threshold 및 5-Stack 요약 설정 매뉴얼
Accuracy Visualizer(시선 측정 정확도 시각화) 플러그인이 실행될 때 적용되는 이상치 각도 차단 기준값(Outlier Threshold [degrees]) 및 5-Stack 통계 시연 옵션 기본값 설정 방법입니다.

- **파일 경로**: [pupil_src/shared_modules/accuracy_visualizer.py](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/accuracy_visualizer.py#L178-L215)
- **클래스 및 함수 위치**: `Accuracy_Visualizer.__init__` (Line 181)

```python
class Accuracy_Visualizer(Plugin):
    def __init__(
        self,
        g_pool,
        outlier_threshold=1.2,  # Outlier Threshold 디폴트값 (기본 1.2도)
        vis_mapping_error=True,
        vis_calibration_area=True,
        enable_5stack_summary=False,  # 5-Stack Validation 통계 요약 콘솔 출력 옵션 (기본 False: 시연 시 UI에서 활성화)
        stack_target_count=5,         # 스택 타겟 횟수 (5회)
    ):
```

- **수정 가이드**:
  - 테스트(Validation) 실행 시 자동으로 초기화되는 Outlier Threshold 기본값을 변경하려면 `outlier_threshold=1.2` 기본 매개변수의 숫자값을 원하는 값(예: `1.5`, `2.0` 등)으로 직접 변경하시면 됩니다.
  - **시연용 5-Stack 요약 기능**: Pupil Capture 실행 후 **World Window 우측 사이드바 > `Accuracy Visualizer` 메뉴** 내의 **`5-Stack Summary Demo Output`** 스위치를 켜면(ON) 캘리브레이션 후 밸리데이션 5회 누적 시 콘솔에 평균/표준편차 배너가 출력됩니다.
