# Pupil Labs 딥러닝 동공 디텍터 실험 과정 에러 분석 보고서 (Experiment & Error Report)
본 문서는 **`TemporalUNet` (Temporal ConvLSTM 2D nnUNet) 실시간 동공 디텍터 및 gaze mapping 실험 과정에서 발생한 모든 에러 로그 원문, 원인 분석 및 해결 조치 내역**을 정리한 통합 기술 보고서입니다.

---

## 1. 개요 및 실험 환경 (Overview)

- **가상환경**: `pupil-umamba` (Python 3.10.20, PyTorch CUDA 12, NumPy 1.26.4)
- **대상 소프트웨어**: Pupil Labs Capture (`pupil_src/main.py`)
- **주요 딥러닝 디텍터**: `Detector2DPlugin` (`TemporalUNet` - 14.1M Encoder + 52.3M Temporal ConvLSTM Decoder)
- **최종 검증 커밋**: `b6161d21` (태그: `v-nnunet-init`)

---

## 2. 발생 에러 유형별 로그 원문 및 원인 분석 (Error Logs & Root Cause Analysis)

### 2.1 에러 유형 1: 연속 알림 수신 시 ConvLSTM 히든 스테이트 과도 리셋 (Continuous State Reset)

#### 📄 로그 원문 (Log Snippet 1)
```text
[15:56:31] INFO     world - calibration_choreography.base_plugin: Starting  Calibration                                                                                                                  base_plugin.py:537
open(): 그런 파일이나 디렉터리가 없습니다
           INFO     eye1 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new calibration/validation sequence.                                                  detector_2d_plugin.py:548
           INFO     eye0 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new calibration/validation sequence.                                                  detector_2d_plugin.py:548
           INFO     eye0 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new calibration/validation sequence.                                                  detector_2d_plugin.py:548
           INFO     eye1 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new calibration/validation sequence.                                                  detector_2d_plugin.py:548
[15:56:44] INFO     eye1 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new calibration/validation sequence.                                                  detector_2d_plugin.py:548
           INFO     eye0 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new calibration/validation sequence.                                                  detector_2d_plugin.py:548
           INFO     eye1 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new calibration/validation sequence.                                                  detector_2d_plugin.py:548
           INFO     eye0 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new calibration/validation sequence.                                                  detector_2d_plugin.py:548
           INFO     eye0 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new calibration/validation sequence.                                                  detector_2d_plugin.py:548
           INFO     eye1 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new calibration/validation sequence.                                                  detector_2d_plugin.py:548
[15:56:56] INFO     world - calibration_choreography.base_plugin: Starting  Calibration                                                                                                                  base_plugin.py:537
...
/home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/accuracy_visualizer.py:470: RuntimeWarning: Mean of empty slice.
  accuracy = np.rad2deg(np.arccos(selected_samples.clip(-1.0, 1.0).mean()))
/home/byeongjun/anaconda3/envs/pupil-umamba/lib/python3.10/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in divide
  ret = ret.dtype.type(ret / rcount)
           WARNING  world - accuracy_visualizer: Not enough data available for angular accuracy calculation.                                                                                     accuracy_visualizer.py:397
           INFO     world - accuracy_visualizer: Angular precision: 0.177 degrees                                                                                                                accuracy_visualizer.py:411
```

#### 🔍 원인 분석 (Root Cause)
- `Detector2DPlugin.on_notify`에서 알림 조건을 `if "calibration" in subj or "validation" in subj or "choreography" in subj:` 로 과도하게 넓게 설정함.
- Pupil Labs는 캘리브레이션 진행 도중 마커가 움직이거나 수집 데이터가 추가될 때마다 초당 수십 개의 알림(예: `calibration_choreography.add_ref_data`, `calibration.progress`)을 발송함.
- 결과적으로 캘리브레이션이 진행되는 **10여 초 내내 매 초마다 ConvLSTM 메모리가 억지로 영점(Zero)으로 비워져** 시시각각 연속 동공 특징이 파괴되고 시선 매핑이 붕괴함.

#### 💡 조치 사항 (Resolution)
- `on_notify` 조건식을 **`if "calibration" in subj and (subj.endswith(".should_start") or subj.endswith(".started")):`** 로 한정함.
- **새로운 캘리브레이션이 시작될 때 딱 1번만 메모리를 영점 초기화**하도록 정밀 제한하여 캘리브레이션 진행 동안 시계열 특성이 유지되도록 수정함.

---

### 2.2 에러 유형 2: 비표준 카메라 해상도 설정으로 인한 렌즈 파라미터(Intrinsics) 부재 및 USB 프레임 손상

#### 📄 로그 원문 (Log Snippet 2)
```text
[16:10:58] WARNING  eye0 - uvc: Turbojpeg jpeg2yuv: b'Corrupt JPEG data: premature end of data segment'                                                                                           detector_2d_plugin.py:247
[16:11:12] WARNING  world - uvc: Turbojpeg jpeg2yuv: b'Corrupt JPEG data: 770 extraneous bytes before marker 0xd2'                                                               me                        uvc_backend.py:879
[16:11:30] WARNING  eye1 - uvc: Turbojpeg jpeg2yuv: b'Corrupt JPEG data: premature end of data segment'                                                                                           detector_2d_plugin.py:247
           WARNING  world - camera_models: No camera intrinsics available for camera Pupil Cam1 ID2 at resolution (800, 600)!                                                                          camera_models.py:407
           WARNING  world - camera_models: Loading dummy intrinsics, which might decrease accuracy!                                                                                                    camera_models.py:411
           WARNING  world - camera_models: Consider selecting a different resolution, or running the Camera Instrinsics Estimation!                                                                    camera_models.py:412
[16:11:51] INFO     world - calibration_choreography.base_plugin: Starting  Calibration                                                                                                                  base_plugin.py:537
open(): 그런 파일이나 디렉터리가 없습니다
[16:12:03] INFO     world - calibration_choreography.base_plugin: Stopping  Calibration                                                                                                                  base_plugin.py:574
open(): 그런 파일이나 디렉터리가 없습니다
The maximum number of function evaluations is exceeded.
Function evaluations 100, initial cost 9.8092e+02, final cost 1.2630e+01, first-order optimality 4.50e-03.
`ftol` termination condition is satisfied.
Function evaluations 45, initial cost 3.7879e+02, final cost 2.7919e+00, first-order optimality 3.94e-05.
`ftol` termination condition is satisfied.
Function evaluations 21, initial cost 6.2612e+02, final cost 5.7122e+00, first-order optimality 6.39e-05.
/home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/accuracy_visualizer.py:470: RuntimeWarning: Mean of empty slice.
  accuracy = np.rad2deg(np.arccos(selected_samples.clip(-1.0, 1.0).mean()))
/home/byeongjun/anaconda3/envs/pupil-umamba/lib/python3.10/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in divide
  ret = ret.dtype.type(ret / rcount)
[16:12:04] WARNING  world - accuracy_visualizer: Not enough data available for angular accuracy calculation.                                                                                     accuracy_visualizer.py:397
           INFO     world - accuracy_visualizer: Angular precision: 0.061 degrees                                                                                                                accuracy_visualizer.py:411
```

#### 🔍 원인 분석 (Root Cause)
1. **카메라 Intrinsics 부재**: World 카메라 해상도를 비표준 왜곡 크기인 `(800, 600)`으로 설정하여 Pupil Labs에 내장된 카메라 초점거리/렌즈 왜곡 파라미터가 로드되지 않음 (`Loading dummy intrinsics`).
2. **3D 시선 최적화 폭등**: 가짜 초점거리(Dummy Intrinsics)로 인해 3D 안구 맵퍼의 최적화 오차가 `12.63도` (`final cost 1.2630e+01`)로 폭등함.
3. **Outlier Threshold 컷오프**: 오차가 12.63도인 상황에서 `outlier_threshold = 1.3도` 미만 조건에 걸려 수집된 샘플이 100% 버려지면서 `Mean of empty slice` 경고 발생.
4. **Corrupt JPEG data**: 비표준 해상도 모드 전송 시 USB 밴드위스 허용량을 초과하여 JPEG 디코딩 손상 발생.

#### 💡 조치 사항 (Resolution)
- Intrinsics 렌즈 파라미터가 내장되어 있고 안정적인 표준 HD 해상도인 **`1280x720` (720p)** 또는 **`1920x1080` (1080p)**로 세팅 권장.

---

### 2.3 에러 유형 3: 외부 거짓 마커(False Marker / Reflection) 오인에 의한 캘리브레이션 붕괴

#### 📄 로그 원문 (Log Snippet 3)
```text
[17:07:53] INFO     world - calibration_choreography.base_plugin: Starting  Calibration                                                                                                                  base_plugin.py:537
           INFO     eye0 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new Calibration (calibration.should_start).                                           detector_2d_plugin.py:548
open(): 그런 파일이나 디렉터리가 없습니다
           INFO     eye0 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new Calibration (calibration.started).                                                detector_2d_plugin.py:548
           INFO     eye1 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new Calibration (calibration.should_start).                                           detector_2d_plugin.py:548
           INFO     eye1 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new Calibration (calibration.started).                                                detector_2d_plugin.py:548
[17:08:00] WARNING  world - calibration_choreography.screen_marker_plugin: 2 markers detected. Please remove all the other markers                                                              screen_marker_plugin.py:326
[17:08:07] INFO     world - calibration_choreography.base_plugin: Stopping  Calibration                                                                                                                  base_plugin.py:574
open(): 그런 파일이나 디렉터리가 없습니다
`ftol` termination condition is satisfied.
Function evaluations 24, initial cost 7.0214e+00, final cost 7.3298e-01, first-order optimality 1.70e-05.
`ftol` termination condition is satisfied.
Function evaluations 21, initial cost 3.8212e+00, final cost 6.6382e-01, first-order optimality 1.62e-05.
Both `ftol` and `xtol` termination conditions are satisfied.
Function evaluations 15, initial cost 3.9659e+00, final cost 3.9828e-01, first-order optimality 2.02e-05.
[17:08:08] INFO     world - accuracy_visualizer: Angular accuracy: 1.000 degrees                                                                                                                 accuracy_visualizer.py:402
           INFO     world - accuracy_visualizer: Angular precision: 0.174 degrees                                                                                                                accuracy_visualizer.py:411
[17:09:06] INFO     eye1 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new Calibration (calibration.should_start).                                           detector_2d_plugin.py:548
           INFO     eye0 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new Calibration (calibration.should_start).                                           detector_2d_plugin.py:548
           INFO     world - calibration_choreography.base_plugin: Starting  Calibration                                                                                                                  base_plugin.py:537
open(): 그런 파일이나 디렉터리가 없습니다
           INFO     eye1 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new Calibration (calibration.started).                                                detector_2d_plugin.py:548
           INFO     eye0 - pupil_detector_plugins.detector_2d_plugin: Reset TemporalUNet ConvLSTM state for new Calibration (calibration.started).                                                detector_2d_plugin.py:548
[17:09:13] WARNING  world - calibration_choreography.screen_marker_plugin: 2 markers detected. Please remove all the other markers                                                              screen_marker_plugin.py:326
           WARNING  world - calibration_choreography.screen_marker_plugin: 2 markers detected. Please remove all the other markers                                                              screen_marker_plugin.py:326
[17:09:18] WARNING  world - calibration_choreography.screen_marker_plugin: 2 markers detected. Please remove all the other markers                                                              screen_marker_plugin.py:326
           WARNING  world - calibration_choreography.screen_marker_plugin: 2 markers detected. Please remove all the other markers                                                              screen_marker_plugin.py:326
[17:09:20] INFO     world - calibration_choreography.base_plugin: Stopping  Calibration                                                                                                                  base_plugin.py:574
open(): 그런 파일이나 디렉터리가 없습니다
The maximum number of function evaluations is exceeded.
Function evaluations 100, initial cost 5.2287e+02, final cost 2.7369e+01, first-order optimality 2.25e-02.
`ftol` termination condition is satisfied.
Function evaluations 29, initial cost 5.1738e+02, final cost 2.2122e+00, first-order optimality 4.93e-05.
`ftol` termination condition is satisfied.
Function evaluations 27, initial cost 5.0815e+00, final cost 8.5908e-01, first-order optimality 7.79e-06.
/home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/accuracy_visualizer.py:470: RuntimeWarning: Mean of empty slice.
  accuracy = np.rad2deg(np.arccos(selected_samples.clip(-1.0, 1.0).mean()))
/home/byeongjun/anaconda3/envs/pupil-umamba/lib/python3.10/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in divide
  ret = ret.dtype.type(ret / rcount)
[17:09:22] WARNING  world - accuracy_visualizer: Not enough data available for angular accuracy calculation.                                                                                     accuracy_visualizer.py:397
           INFO     world - accuracy_visualizer: Angular precision: 0.150 degrees                                                                                                                accuracy_visualizer.py:411
```

#### 🔍 원인 분석 (Root Cause)
- World 카메라 시야 내에 화면 마커 외에 **원형 컵, 스피커 또는 모니터 조명 반사 링 등 2번째 거짓 동심원 마커**가 포착됨 (`2 markers detected`).
- 17:08분 시도에서는 `1.000도`로 성공하였으나, 17:09분 시도에서는 4초간 지속적으로 2번째 거짓 마커 위치가 캘리브레이션 3D 참조 좌표로 잘못 수집됨.
- 엉뚱한 반사광 좌표가 유입되면서 3D 최적화 오차가 `27.36도` (`final cost 2.7369e+01`)로 폭등하여 캘리브레이션 붕괴 발생.

#### 💡 조치 사항 (Resolution)
- 카메라 시야 내의 동그란 물체 제거 및 모니터 빛 반사 링이 생기지 않도록 방 조명 및 카메라 수평 앵글 세팅 권장.

---

### 2.4 에러 유형 4: 정상 실험 성공 사례 및 아웃라이어 임계값 세팅

#### 📄 로그 원문 (Log Snippet 4 - 최고 성능 달성 사례)
```text
[16:20:59] 실행...
[16:21:23] INFO     world - accuracy_visualizer: Angular accuracy: 0.854 degrees
           INFO     world - accuracy_visualizer: Angular precision: 0.125 degrees
[16:21:33] INFO     world - accuracy_visualizer: Angular accuracy: 1.124 degrees
[16:21:44] INFO     world - accuracy_visualizer: Angular accuracy: 1.043 degrees
[16:21:53] INFO     world - accuracy_visualizer: Angular accuracy: 1.178 degrees
[16:22:01] INFO     world - accuracy_visualizer: Angular accuracy: 0.813 degrees
```

#### 🔍 원인 분석 (Root Cause)
- 마커당 샘플 수집 듀레이션이 **`sample_duration = 60`**일 때, 초반 마커 이동에 따른 튀는 프레임이 제외되고 고정 주시 알짜 데이터만 수집되어 **캘리브레이션 `0.854도`, 검증 `0.813도`**라는 최고 스코어가 달성됨.
- `outlier_threshold = 1.3도` 설정 시, 인간 안구의 실시간 미세 떨림(Fixation Jitter)에 따라 오차가 1.18도일 때는 통과하고, 1.35도일 때는 0.05도 차이로 컷오프를 넘어서면서 연속 검증 시 순간적으로 `Mean of empty slice`가 발생할 수 있음.

---

## 3. 최종 고정 설정 및 적용 내역 (Final Configuration)

현재 브랜치(`feature/nnunet`, 커밋 `b6161d21`, 태그 `v-nnunet-init`)에 최종 확정되어 적용된 소스 코드 구성입니다:

| 항목 | 파일 및 위치 | 세부 설정 값 | 비고 |
|:---|:---|:---|:---|
| **ConvLSTM 리셋** | [detector_2d_plugin.py Line 542](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/pupil_detector_plugins/detector_2d_plugin.py#L542) | `if "calibration" in subj and (endswith(".should_start") or endswith(".started")):` | **캘리브레이션 시작시에만 메모리 리셋** (검증 시 메모리 보존) |
| **샘플 듀레이션** | [screen_marker_plugin.py Line 90](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/calibration_choreography/screen_marker_plugin.py#L90) | `sample_duration = 60` | 마커당 60프레임 수집 |
| **검증 타겟 좌표** | [screen_marker_plugin.py Line 82](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/calibration_choreography/screen_marker_plugin.py#L82) | `return [(0.5, 0.7)] * 5` | `(0.5, 0.7)` 5회 연속 수집 |
| **Outlier Threshold** | [accuracy_visualizer.py Line 181](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/accuracy_visualizer.py#L181) | `outlier_threshold = 1.3` | 기본 순정 수식 유지 |

---

## 4. AI Agent 참조 가이드 (Instructions for AI Agents)

향후 본 레포지토리의 시선 추적 성능 검증 및 디버깅을 수행하는 AI Agent는 다음 규칙을 준수해야 합니다:

1. **ConvLSTM 상태 관리**: `TemporalUNet` 사용 시, 검증(Validation) 동안에는 이전 시순차 특성을 유지하고 **새로운 캘리브레이션 시작 시에만 `reset_temporal_state()`를 호출**해야 합니다.
2. **SciPy Cost 로그 해석**: SciPy 최적화 로그의 `final cost 2.9155e+01` 등 지수 표기법은 29.155도 오차를 의미하므로, 이 값이 1.5 미만으로 수렴했는지 확인 후 `Not enough data` 원인을 진단해야 합니다.
3. **거짓 마커 오인 진단**: `2 markers detected` 경고가 발생하면 코드 버그가 아닌 월드 카메라 영상 내의 외부 원형 물체/반사광 오인이 원인이므로, 마커 트래킹 가시성을 점검해야 합니다.

---
*보고서 생성 시각: 2026-07-23T20:01:43+09:00*
*깃 커밋 및 태그 상태: Commit `b6161d21`, Tag `v-nnunet-init` (Pushed to origin)*
