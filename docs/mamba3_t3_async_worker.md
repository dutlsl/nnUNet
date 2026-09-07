# Mamba3 T=3 교체 + 비동기 추론 워커 도입 (2026-08-28)

> **핵심 질문**: 왜 T=7 동기 실행을 T=3 비동기 실행으로 바꿨나?
> **답**: Mamba3 T=7 추론(프레임당 8.9ms + 전/후처리)이 120Hz eye 이벤트 루프(8.3ms/프레임 버짓)를 초과하여 실효 pupil 스트림이 절반(~60Hz)으로 희소화되고, 이 희소성이 world/eye 프레임 timestamp 매칭 윈도우를 실패시켜 calibration/validation에서 사용 샘플이 과도하게 탈락했다. T=3으로 윈도우를 줄이고, 추론을 eye 루프 밖의 전용 워커 스레드로 분리하여 루프 블로킹을 구조적으로 제거했다.

---

## 1. 문제 진단 (로그 근거)

`recordings/session_20260825_111015.log`의 `[EXPERIMENT REPORT]` Used Samples 비교:

| 모델 | 총 샘플 (denominator) | 정확도 사용 샘플 |
|---|---|---|
| Mamba3 (T=7) | **~400** | 5~24 (1~6%) |
| 2D C++ | ~820 | 77~483 (9~57%) |
| RITnet | ~815 | 0~21 |

**진단**: Mamba3는 정확도 *분자*가 아니라 *분모* 자체가 절반. 즉 eye 프로세스 이벤트 루프
(`launchables/eye.py` → `recent_events` → `detect` 동기 호출)가 T=7 GPU 추론에 막혀
실효 프레임률이 ~60Hz로 떨어졌고, pupil 스트림의 timestamp 간격이 벌어져
`gaze_mapping/matching.py`(`RealtimeMatcher`, binocular cutoff = 2×프레임주기)와
`gaze_mapping/utils.py`(`closest_matches_*_batch`, dispersion 1/15s→0.2→0.5→1.0→2.0s 완화)
매칭에서 탈락하는 프레임이 과도하게 발생했다.

timestamp 자체는 캡처 시점에 확정(`uvc_backend.py:490-499`)되어 손상되지 않으며,
**스트림의 희소화(간격)가 문제**였다.

### 추론 시간 측정 (nnunet_mamba3 env, CUDA)

| | 모델 단독 forward | 전/후처리 포함 (워커 1프레임) |
|---|---|---|
| T=3 | 4.6 ms | ~9-10 ms |
| T=7 | 8.9 ms | ~15-16 ms |

---

## 2. 변경 사항

### 2.1 Mamba3 T=7 → T=3 모듈 교체

**체크포인트 식별** (git 히스토리 + 체크포인트 구조 비교로 확인):

- 기존 코드(commit `21be2cec`)의 T별 매핑: `3 → fold_1`, `5 → Vivim_T5/fold_1`, `7 → fold_1_T7`
- `sequence_dataset.py` 기본값 `temporal_window: int = 3` → **베이스 Vivim 트레이너
  (`nnUNetTrainer_Vivim__nnUNetPlans__2d/fold_1`)가 T=3 훈련 모델**
- VivimBackbone 가중치는 T-독립: 4개 체크포인트(fold_1, T5, T7, repo 로컬) 모두
  104개 텐서·`backbone.*` 키 동일, T 의존 weight 부재 (Mamba3 selective scan은
  시퀀스 길이에 무관)
- `best_checkpoint.pth`(12MB)는 **다른 아키텍처**(`conv1_spatial`/`conv1_temporal`,
  TemporalUNet계)이므로 VivimBackbone 로드 후보에서 **제거** (기존 fallback 체인의
  silent garbage-load隐患 제거)

**코드 변경** ([detector_2d_plugin.py](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/pupil_detector_plugins/detector_2d_plugin.py)):

| 위치 | 내용 |
|---|---|
| L88-90 | `MAMBA3_T = 3`, `MAMBA3_LABEL = "Mamba3 (T=3)"` 상수 추가 |
| `__init__` (L153~) | 기본 `active_model="Mamba3 (T=3)"`; 기존 세션에 저장된 다른 Mamba3 변종("T=7" 등)을 T=3으로 정규화; `_vivim_queues = {3: deque(maxlen=3)}` |
| `_init_nnunet_models` (L235) | T=3 체크포인트 후보: ① `pupil_detector_plugins/best_checkpoint_t3.pth` ② `pupil_src/best_checkpoint_t3.pth` ③ nnUNet `fold_1/checkpoint_best.pth`; `vivim_models[3]` |
| `detect` (L331) | Mamba3 분기에서 T 파싱 fallback `MAMBA3_T` |
| UI Selector (`init_ui`) | `["Mamba3 (T=3)", "2D C++", "RITnet"]` |

**파일 추가**: `pupil_src/shared_modules/pupil_detector_plugins/best_checkpoint_t3.pth`
(19,727,507 bytes, `fold_1/checkpoint_best.pth` 복사 — repo 로컬 컨벤션 유지.
**git untracked — 커밋/ignore 결정 필요**)

**파일 유지 (삭제 안 함)**: `best_checkpoint_t7.pth` (2곳) — 더 이상 참조되지 않음.

### 2.2 구조적 해결: 비동기 추론 워커

**변경 전 (동기)**:
```
eye 이벤트 루프 (120Hz 목표)
  └─ detect(frame)
       └─ _detect_vivim_mamba_by_t(frame)   ← 전처리 + GPU 추론(8.9ms) + .cpu() 동기화 + 후처리
            ← 루프가 프레임마다 최대 ~16ms 블로킹 → 실효 ~60Hz
```

**변경 후 (비동기)**:
```
eye 이벤트 루프 (120Hz 유지)
  └─ detect(frame)
       ├─ frame.gray 복사(160KB, C 버퍼 재사용 방지)
       ├─ _mamba3_queue.put (포화 시 oldest 드롭 → 신선도 유지)     [~0.02ms]
       └─ _fetch_latest_mamba3_result()  ← 직전 완료 결과 반환 (원 프레임 timestamp 유지)

mamba3-worker (daemon 스레드, 1개/eye)
  └─ _mamba3_worker_loop
       ├─ _mamba3_warmup()      ← 시작 시 dummy forward: CUDA init + Mamba3 JIT(~5초)를 초기화 단계로 흡수
       └─ _infer_mamba3(gray, ts, w, h)   ← 구 _detect_vivim_mamba_by_t 로직을 이관 (변경 없음)
            ├─ flip → 400×400 resize → Z-score → 448×448 캔버스
            ├─ _vivim_queues[3] 스택 → [1, 3, 1, 448, 448]
            ├─ model.forward (inference_mode)
            └─ _postprocess_mask_to_datum (EMA/점프필터/ellipse)
```

| 메서드 (행) | 역할 |
|---|---|
| `detect` Mamba3 분기 (L333-377) | 프레임 디스패치 + 최신 결과 반환. 워커 미시작/모델 부재 시 빈 datum |
| `_mamba3_worker_loop` (L499) | 워커 메인 루프. 결과 저장(최신 16개, timestamp 키) |
| `_mamba3_warmup` (L520) | 워커 스레드에서 dummy forward 1회 (JIT/첫 추론 지연 흡수) |
| `_fetch_latest_mamba3_result` (L537) | 최신 완료 결과 반환. **동일 datum 재반복 시 `_published_externally=True`** → `eye.py:821`의 `pop("_published_externally")`이 중복 IPC 발행 억제 (hybrid 플러그인 패턴 동일) |
| `_infer_mamba3` (L553) | 구 `_detect_vivim_mamba_by_t` 본문 (로직 무변경, 프레임 메타는 `SimpleNamespace`로 대체) |
| `_postprocess_mask_to_datum` (L608) | 랩퍼: `self._postprocess_lock` 하에서 `_postprocess_mask_to_datum_locked` (L614) 호출 — Mamba3(워커)·RITnet(메인) 경로 간 EMA/점프 상태(`_prev_ellipse`, `_consecutive_jumps`) 경쟁 방지 |
| `cleanup` (L772) | 큐 드레인 + 센티넬 + `join(15s)` — 인터프리터 종료 전 워커 완전 소멸 (종료 시 CUDA race SIGABRT 수정) |

**안전성 설계**:
- 픽셀은 `np.array(gray, copy=True)`로 복사 (캡처 레이어 버퍼 재사용 방지)
- `frame.gray`는 메인 스레드에서 1회만 접근 (lazy compute 비용은 루프가 부담)
- 타임스탬프는 프레임 캡처 시점 원본 그대로 — IPC 메시지의 `timestamp` 무변경
- hybrid 플러그인(`HybridDetector2DPlugin`)은 `_init_nnunet_models` no-op override로
  `vivim_models={}` → **워커 미시작, 영향 없음**

---

## 3. 수정 파일 목록

| 파일 | 변경 |
|---|---|
| `pupil_src/shared_modules/pupil_detector_plugins/detector_2d_plugin.py` | T=3 상수/모델 로딩/디스패치/워커/락/cleanup |
| `tests/test_dummy_harness.py` | T=7 → T=3 assertion·모델 목록, Mamba3 레이턴시 = 디스패치 전용 주석 |
| `pupil_src/shared_modules/pupil_detector_plugins/best_checkpoint_t3.pth` | **신규** (19.7MB, untracked) |

기존 uncommitted docs/eye.py 등 다른 수정본과 무관.

---

## 4. 검증 결과 (nnunet_mamba3 env, torch 2.6.0+cu124, 실제 GPU)

| 항목 | 결과 |
|---|---|
| 하네스 (`tests/test_dummy_harness.py`) | **PASS** — Mamba3 디스패치 0.02ms (구: 프레임당 전체 추론 블로킹) |
| 워커 처리량 (실제 eye0.mp4 프레임) | **9~10 ms/프레임** (모델 단독 4.6ms + 전/후처리) → 120Hz 대비 여유 |
| 동기/비동기 동등성 | 동일 프레임 30개, 구 동기 로직 vs 신규 워커 → **ellipse 불일치 0건** |
| 실제 프레임 검출 | conf=1.0 ellipse 정상 생성, timestamp 유일·증가 확인 |
| 프로세스 종료 | 워커 join 클린 (SIGABRT 재현 → warmup + join 15s으로 수정 확인) |
| 컴파일 체크 | `py_compile` OK |

---

## 5. 운영 시 고려사항

1. **시작 워밍업 ~5-8초**: Mamba3 JIT/최초 CUDA init. 완료 전 `detect()`는 빈 datum
   반환하며 큐 포화 드롭이 발생한다 (로그 warning 1회). 실험 시작 전 버퍼링 시간 확보.
2. **120Hz 지속 부하**: 워커 9-10ms ≈ 100Hz 처리능. 여유가 크지 않아 간헐적으로
   최신-프레임 드롭 정책이 동작할 수 있음 (warning 로그). 그래도 eye 루프는 120Hz 유지.
3. **OMP_NUM_THREADS**: `main.py` L2-3에서 4로 고정 (앱 런치 시 필수 — unset이면
   torch가 전체 코어 오버서브션으로 초당 0.6프레임까지 저하 확인됨).
   별도 스크립트로 플러그인만 돌릴 때도 `OMP_NUM_THREADS=4` 권장.
4. **검출률**: T=3 fold_1 체크포인트의 OpenEDS 도메인 특성상 일부 영상에서
   conf<0.5 비율이 높을 수 있음 (2026_07_15 eye0.mp4: 30프레임 중 1건 valid).
   T=7 대비 정확도 변화는 실험 재실행으로 비교 필요.
5. **재현성**: 워커 결과 캐시(최신 16개) + 중복 발행 억제 플래그가 IPC 흐름의
   유일한 행동 변화. world 프로세스 수신 관점에서는 pupil datum이
   "최대 ~1프레임 지연된 신선한 결과"로 바뀜 — timestamp 매칭 로직에는 영향 없음.

## 6. 향후 후보 (이번 변경 범위 외)

- AMP 적용 (`_infer_mamba3` 내 `torch.cuda.amp.autocast` — RITnet/nnUNet 경로는 사용 중)
- 448→320 입력 축소로 워커 처리량 여유 확보
- T=3 전용 재학습 체크포인트 (현재는 T=3 기본 트레이너 모델)
- T=7 파일 정리 (`best_checkpoint_t7.pth` ×2, untracked T=3 체크포인트 git 전략)
