# Pupil Labs 시스템 프로세스 간 커뮤니케이션 다이어그램 (Communication & Interaction Diagram)

> **문서 목적**: Pupil Labs 멀티프로세스 아키텍처(`main`, `eye`, `world`) 간의 ZMQ IPC 메시징, 노티피케이션 버스, 동공 검출, 시선 매핑, 캘리브레이션 및 정확도 검증 파이프라인의 **실시간 상호작용 및 통신 흐름**을 표준 커뮤니케이션 다이어그램(Communication Diagram) 및 시퀀스 흐름으로 명세함.
> **연관 문서**: [버전 관리 보고서 (version_management.md)](file:///home/byeongjun/PycharmProjects/pupil/docs/version_management.md)
> **최종 갱신일**: 2026-08-20

---

## 1. 전체 시스템 커뮤니케이션 다이어그램 (System Communication Diagram)

시스템 구성요소(프로세스 및 주요 모듈) 간의 연결 링크와 메시지 번호 체계(`1.1` ~ `7.3`)를 나타낸 시스템 커뮤니케이션 아키텍처 맵입니다.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             1. Main Orchestrator Process (main.py)                              │
│                                                                                                  │
│  ┌───────────────────────┐                    ┌───────────────────────────────────────────────┐  │
│  │     Main Launcher     │───[ 1.1: spawn ]──▶│              ZMQ IPC Backbone                 │  │
│  │ (Process & Lifecycle) │◀──[ 1.2: notify ]──│   (XPUB: 50020 / XSUB: 50021 / PULL: 50022)   │  │
│  └───────────────────────┘                    └───────────────────────┬───────────────────────┘  │
│                                                                       │                          │
│  ┌──────────────────────────────────────────────────────────────┐     │                          │
│  │ EphemeralLogTee (stdout/stderr -> pupil_capture.log)        │◀────┘ (Console Log Mirror)      │
│  └──────────────────────────────────────────────────────────────┘                                │
└───────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                │
                 ▲                              │                              ▲
                 │ [2.3: PUB 'pupil.0.2d']      │ [3.1: SUB 'pupil.*']         │
                 │ [7.1: SUB 'calibration.*']   │ [3.4: PUB 'gaze.2d.*']       │ [5.1/5.2: notify_all]
                 │                              │ [6.1: SUB 'validation.*']    │
                 ▼                              │                              ▼
┌───────────────────────────────────────────────┴──┐  ┌────────────────────────────────────────────┐
│      2. Eye Process (eye.py - eye0 / eye1)       │  │          3. World Process (world.py)       │
│                                                  │  │                                            │
│  ┌────────────────────────────────────────────┐  │  │  ┌──────────────────────────────────────┐  │
│  │ Eye Camera Backend (192x192 / 120 FPS)     │  │  │  │ World Camera Backend (1280x720 / 60) │  │
│  └─────────────────────┬──────────────────────┘  │  │  └──────────────────┬───────────────────┘  │
│                        │ [2.1: frame.gray]       │  │                     │ [4.3: ref_list]      │
│                        ▼                         │  │                     ▼                      │
│  ┌────────────────────────────────────────────┐  │  │  ┌──────────────────────────────────────┐  │
│  │ Detector2DPlugin (Mamba3 / RITnet / 2D C+) │  │  │  │ CalibrationChoreographyPlugin       │  │
│  │ • Preprocess (Z-Score, Letterbox)          │  │  │  │ • Render Target (5p/9p/12p / Diamond)│  │
│  │ • Inference (VivimBackbone / DenseNet)     │  │  │  │ • Collect calib_data {pupil, ref}   │  │
│  │ • Postprocess (Blur, EMA α=0.4, Jump-rej)  │  │  │  └──────────────────┬───────────────────┘  │
│  └─────────────────────┬──────────────────────┘  │  │                     │ [5.1/5.2: calib/val] │
│                        │ [2.2: datum dict]       │  │                     ▼                      │
│                        ▼                         │  │  ┌──────────────────────────────────────┐  │
│  ┌────────────────────────────────────────────┐  │  │  │ Pupil_Data_Relay (ZMQ SUB)           │  │
│  │ Pupil Pub Socket (ZMQ PUB -> IPC Backbone) │  │  │  │ • Receive 'pupil.0.2d'               │  │
│  └────────────────────────────────────────────┘  │  │  └──────────────────┬───────────────────┘  │
│                                                  │  │                     │ [3.2 / 4.2: pupil]   │
│  ┌────────────────────────────────────────────┐  │  │                     ▼                      │
│  │ Threaded Log Parser                        │  │  │  ┌──────────────────────────────────────┐  │
│  │ • [7.2] Read pupil_capture.log (Reverse)   │  │  │  │ Gazer2D / GazerBase                  │  │
│  │ • Extract Accuracy, Precision, RMSE        │  │  │  │ • filter_pupil_data (3D exclusion)   │  │
│  │ • [7.3] Save recordings/*.log              │  │  │  │ • LinearRegression (Fit / Predict)   │  │
│  └────────────────────────────────────────────┘  │  │  └──────────────────┬───────────────────┘  │
│                                                  │  │                     │ [3.3: gaze_datum]    │
│                                                  │  │                     ▼                      │
│                                                  │  │  ┌──────────────────────────────────────┐  │
│                                                  │  │  │ Gaze Pub Socket (ZMQ PUB)            │  │
│                                                  │  │  └──────────────────────────────────────┘  │
│                                                  │  │                                            │
│                                                  │  │  ┌──────────────────────────────────────┐  │
│                                                  │  │  │ Accuracy_Visualizer                  │  │
│                                                  │  │  │ • [6.2] calc_acc_prec_errlines       │  │
│                                                  │  │  │ • Dispersion relaxation (0.2s..2.0s) │  │
│                                                  │  │  │ • Outlier filter (1.2 deg threshold) │  │
│                                                  │  │  │ • 5-Stack Validation Summary Console │  │
│                                                  │  │  │ • Print "Angular accuracy: X.XXX"    │  │
│                                                  │  │  └──────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘  └────────────────────────────────────────────┘
```

---

## 2. 채널별 프로토콜 및 메시지 스펙 (IPC Message Specification)

### 2.1 ZMQ IPC 토폴로지 구조

| 소켓 이름 | 소켓 유형 | 연결 URL / 엔드포인트 | 메시지 패턴 | 전송 데이터 형식 |
|:---|:---|:---|:---|:---|
| **IPC Pub Proxy (XPUB)** | `zmq.XPUB` | `ipc:///tmp/pupil_ipc_pub` / `tcp://127.0.0.1:50020` | Publisher 브로드캐스트 | Topic + MsgPack Payload |
| **IPC Sub Proxy (XSUB)** | `zmq.XSUB` | `ipc:///tmp/pupil_ipc_sub` / `tcp://127.0.0.1:50021` | Subscriber 메시지 집약 | Topic + MsgPack Payload |
| **Push-Pull Bridge** | `zmq.PULL` | `ipc:///tmp/pupil_ipc_push` / `tcp://127.0.0.1:50022` | 비동기 Push 알림 수신 | MsgPack Notification dict |

---

### 2.2 주요 메시지 토픽 및 Payload 스펙

#### 1) 동공 검출 데이터 (`pupil.<eye_id>.<method>`)
- **발행처**: Eye Process (`Detector2DPlugin` → `pupil_socket`)
- **수신처**: World Process (`Pupil_Data_Relay` → `pupil_sub`)
```json
{
  "topic": "pupil.0.2d",
  "id": 0,
  "method": "Mamba3 (T=7)",
  "norm_pos": [0.5234, 0.4812],
  "diameter": 45.2,
  "confidence": 0.892,
  "timestamp": 1724138120.456,
  "ellipse": {
    "axes": [45.2, 41.8],
    "angle": 14.5,
    "center": [98.2, 91.4]
  }
}
```

#### 2) 시선 좌표 데이터 (`gaze.2d.<eye_id>.`)
- **발행처**: World Process (`Gazer2D` → `gaze_pub`)
- **수신처**: IPC Backbone (모든 플러그인 및 외부 ZMQ 수신자)
```json
{
  "topic": "gaze.2d.0.",
  "norm_pos": [0.5120, 0.6980],
  "confidence": 0.892,
  "timestamp": 1724138120.456,
  "base_data": [{ "topic": "pupil.0.2d", ... }]
}
```

#### 3) 캘리브레이션/검증 알림 메시지 (Choreography Notification)
- **발행처**: World Process (`CalibrationChoreographyPlugin` → `notify_all()`)
- **수신처**: `Accuracy_Visualizer`, `Gazer2D`, `Detector2DPlugin`
```json
{
  "subject": "calibration_choreography.data",
  "mode": "validation",
  "action": "data",
  "gazer_class_name": "Gazer2D",
  "gazer_params": {
    "left_model": { ... },
    "right_model": { "coef_": [...], "intercept_": [...] },
    "enable_calibration": true
  },
  "pupil_list": [{ "norm_pos": [...], "timestamp": ... }],
  "ref_list": [{ "norm_pos": [0.5, 0.7], "timestamp": ... }],
  "timestamp": 1724138130.123,
  "record": true
}
```

---

## 3. 핵심 파이프라인 시퀀스 상호작용 흐름 (Interaction Flow)

### 3.1 캘리브레이션(Calibration) 실행 시 통신 흐름

```
[User]          [Choreography]        [Eye Process]        [IPC Backbone]       [Gazer2D]        [AccuracyViz]       [Log File]
  │                   │                     │                    │                  │                  │                  │
  ├── 1. Click 'C' ──▶│                     │                    │                  │                  │                  │
  │                   ├── 2. notify_all("calibration.started") ─▶│                  │                  │                  │
  │                   │                     │◀── 3. Dispatch ────┤                  │                  │                  │
  │                   │                     │ (Reset ConvLSTM 1x)│                  │                  │                  │
  │                   │                     │                    │                  │                  │                  │
  │                   │  [ 4. Loop: 9-point Grid Marker Collection (sample_duration=60 frames) ]       │                  │
  │                   │  • Eye Process ───▶ PUB 'pupil.0.2d' ──▶ IPC Backbone ──▶ Choreography Buffer │                  │
  │                   │  • ScreenMarkerPlugin renders 3x3 Grid & buffers World Camera ref_list          │                  │
  │                   │                     │                    │                  │                  │                  │
  │                   │                     │                    │                  │                  │                  │
  ├── 5. Click 'C' ──▶│                     │                    │                  │                  │                  │
  │   (Stop Calib)    ├── 6. Start Gazer2D(calib_data={ref_list, pupil_list}) ─────▶│                  │                  │
  │                   │                     │                    │                  │ (fit models)     │                  │
  │                   │                     │                    │◀── 7. notify ────┤                  │                  │
  │                   │                     │                    │ ("calibration.result", params)      │                  │
  │                   │                     │                    ├────────────────────────────────────▶│                  │
  │                   │                     │                    │                  │                  │ (recalculate)    │
  │                   │                     │                    │                  │                  ├── 8. Log Info ──▶│
  │                   │                     │                    │                  │                  │ ("Accuracy: 0.8")│
  │                   │                     │                    │◀── 9. notify ────┤                  │                  │
  │                   │                     │                    │ ("calibration.successful")          │                  │
  │                   │                     │◀── 10. Dispatch ───┤                  │                  │                  │
  │                   │                     │ (Threaded Worker)  │                  │                  │                  │
  │                   │                     ├── 11. Reverse Log Scan (pupil_capture.log) ────────────────────────────────▶│
  │                   │                     ├── 12. Save recordings/Mamba3_T=7_calibration_*.log                          │
```

---

### 3.2 밸리데이션(Validation / Accuracy Test) 실행 시 통신 흐름

```
[User]          [Choreography]        [Eye Process]        [IPC Backbone]       [Gazer2D]        [AccuracyViz]       [Log File]
  │                   │                     │                    │                  │                  │                  │
  ├── 1. Click 'T' ──▶│                     │                    │                  │                  │                  │
  │                   ├── 2. notify_all("validation.started") ──▶│                  │                  │                  │
  │                   │                     │◀── 3. Dispatch ────┤                  │                  │                  │
  │                   │                     │ (Preserve State)   │                  │                  │                  │
  │                   │                     │                    │                  │                  │                  │
  │                   │  [ 4. Loop: Diamond / 4-Corner Marker Collection (sample_duration=60) ]        │                  │
  │                   │  • Eye Process ───▶ PUB 'pupil.0.2d' ──▶ IPC Backbone ──▶ Choreography Buffer │                  │
  │                   │  • ScreenMarkerPlugin renders validation target & buffers new ref_list          │                  │
  │                   │                     │                    │                  │                  │                  │
  │                   │                     │                    │                  │                  │                  │
  ├── 5. Click 'T' ──▶│                     │                    │                  │                  │                  │
  │   (Stop Valid)    │ (Get params from active gazer)           │                  │                  │                  │
  │                   ├── 6. notify_all("validation.data", gazer_params, new_pupil, new_ref) ─────────▶│                  │
  │                   │                     │                    ├────────────────────────────────────▶│                  │
  │                   │                     │                    │                  │                  │ (clear & update) │
  │                   │                     │                    │                  │◀── 7. Map Gaze ──┤                  │
  │                   │                     │                    │                  │ (predict pos)    │                  │
  │                   │                     │                    │                  ├─────────────────▶│                  │
  │                   │                     │                    │                  │                  │ • Relax Dispers  │
  │                   │                     │                    │                  │                  │ • Outlier 1.2 deg│
  │                   │                     │                    │                  │                  │ • 5-Stack check  │
  │                   │                     │                    │                  │                  ├── 8. Log Info ──▶│
  │                   │                     │                    │                  │                  │ ("Accuracy: 1.2")│
  │                   ├── 9. notify_all("validation.stopped") ──▶│                  │                  │                  │
  │                   │                     │◀── 10. Dispatch ───┤                  │                  │                  │
  │                   │                     │ (Threaded Worker)  │                  │                  │                  │
  │                   │                     ├── 11. Reverse Log Scan (pupil_capture.log) ────────────────────────────────▶│
  │                   │                     ├── 12. Save recordings/Mamba3_T=7_test_*.log                                 │
```

---

### 3.3 캘리브레이션 ON / OFF 토글 변경 시 통신 흐름

```
[User]          [Eye UI Plugin]       [IPC Backbone]       [Choreography]       [Gazer2D (World)]   [AccuracyViz]
  │                   │                    │                     │                     │                  │
  ├── 1. Toggle OFF ─▶│                    │                     │                     │                  │
  │   (Switch = False)│ (g_pool.enable=F)  │                     │                     │                  │
  │                   ├── 2. notify_all("calibration.set_enabled", enabled=False) ────▶│                  │
  │                   │                    │                     │                     │                  │
  │                   │                    ├── 3. Dispatch ─────▶│ (g_pool.enable=F)   │                  │
  │                   │                    ├──────────────────────────────────────────▶│ (gazer.enable=F)│
  │                   │                    ├─────────────────────────────────────────────────────────────▶│
  │                   │                    │                     │                     │                  │ • g_pool.enable=F
  │                   │                    │                     │                     │                  │ • gazer_params=F
  │                   │                    │                     │                     │                  │ • recalculate()
  │                   │                    │                     │                     │                  │
  │                   │                    │                     │ [ Real-time Gaze Mapping Behavior ]    │
  │                   │                    │                     │ • predict() bypasses LinearRegression │
  │                   │                    │                     │ • yields raw pupil_norm_pos as gaze   │
```

---

## 4. 커뮤니케이션 병목 지점 및 안정화 조치 내역

| 커뮤니케이션 경로 | 잠재적 병목 및 오류 요인 | 적용된 해결 조치 및 패치 |
|:---|:---|:---|
| **Eye → IPC (`pupil.0.2d`)** | Mamba-3 추론 지연(~50ms)으로 인한 프레임 레이트 저하 및 타임스탬프 불일치 | [`utils.py`](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/gaze_mapping/utils.py)에 **점진적 허용 분산 완화(`max_dispersion: 66ms → 0.2s → 0.5s → 1.0s → 2.0s`)** 적용 |
| **World Process 내부 (`filter_pupil_data`)** | `method` 필드에 `"2d"`가 없는 딥러닝 모델 데이터 전량 드롭 | [`gazer_2d.py`](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/gaze_mapping/gazer_2d.py) 필터를 `"3d" not in method.lower()`로 반전 패치 |
| **Eye ↔ World 노티피케이션 버스** | 초당 수십 회 발생하는 choreography 세부 알림으로 인한 ConvLSTM 무차별 리셋 | [`detector_2d_plugin.py`](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/pupil_detector_plugins/detector_2d_plugin.py)에서 **`calibration.should_start`/`started` 시점에만 1회 리셋**으로 제한 |
| **Eye 프로세스 ↔ 로그 파일** | Eye 프로세스에서 World 프로세스의 Accuracy 결과 직접 취득 불가 | `EphemeralLogTee` 기반 stdout 미러링 및 데몬 스레드 로그 파서 연동 |
| **IPC ↔ Accuracy Visualizer** | 캘리브레이션 ON/OFF 토글 상태가 밸리데이션 재계산 시 누락 | `on_notify`에서 `recent_input.gazer_params["enable_calibration"]` 즉시 갱신 및 `recalculate()` 트리거 |
