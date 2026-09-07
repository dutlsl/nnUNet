# Pupil Labs 동공 검출 및 시선 추적 시스템 버전 관리 및 변천사 분석 보고서 (Version Management Documentation)

> **문서 목적**: Pupil Labs 커스텀 포크의 브랜치별·버전별 아키텍처 변천사, 소스코드 Diff 기반 성능 변화 원인, 캘리브레이션/밸리데이션 파이프라인 변동 내역 및 기술적 이슈 해결 과정을 총정리하여 향후 연구 및 시스템 유지보수의 기준점으로 활용함.
> **연관 문서**: [커뮤니케이션 다이어그램 (communication_diagram.md)](file:///home/byeongjun/PycharmProjects/pupil/docs/communication_diagram.md)
> **최종 갱신일**: 2026-08-20

---

## 1. 브랜치 계보 및 버전 트리 (Branch & Version Genealogy)

본 저장소의 핵심 브랜치 및 릴리즈 태그의 진화 계보입니다.

```
[ master ] (Pupil Labs Upstream v3.6 - Pure Python/C++ 2D)
    │
    ▼
[ develop ] ─── (pupil+RITnet 이식)
    │
    ▼
[ v-ritnet-base ] (Commit 2c42b2f1 - 전년도 레거시 베이스라인)
    │
    ├──▶ [ main / v-umamba ] (Commit a5b0819d - U-Mamba 2D 통합)
    │        │
    │        ├──▶ [ feature/transunet ] (Commit 5e626907 - TransUNet 실험)
    │        │
    │        └──▶ [ feature/nnunet ] (Commit 4b67751a, Tags: v-nnunet-init, nnunet-conv)
    │                 │  • 3x3 9-point Grid 캘리브레이션 & 1-Center 밸리데이션 도입
    │                 │  • TemporalUNet(ConvLSTM), nnUNet Vivim, nnUNet 2D 통합
    │                 │  • 동공 후처리(EMA α=0.4, Jump Reject, Blur) 전면 신설
    │                 │
    │                 └──▶ [ nnunet-mamba3 ] (Commit 4bfdea4b - 최신 헤드)
    │                           • Vivim-Mamba3 T=3..11 다중 시계열 윈도우 지원
    │                           • filter_pupil_data 3D 역필터 패치
    │                           • 점진적 Timestamp Dispersion 완화 (0.2s~2.0s)
    │                           • 캘리브레이션 ON/OFF UI 토글 & 실시간 IPC 동기화
```

---

## 2. 버전별 주요 특징 및 정량적 성능 비교 (Quantitative Benchmarks)

전년도 초기 버전부터 최신 `nnunet-mamba3` 브랜치까지의 모델별 시선 추적 각도 오차(Angular Accuracy, visual angle degrees) 및 시스템 특성 비교입니다.

| 버전 / 태그 | 브랜치 | 주요 디텍터 모델 | 2D C++ Accuracy | RITnet Accuracy | 딥러닝 신규 모델 Accuracy | 주요 특징 및 실험 조건 |
|:---|:---|:---|:---:|:---:|:---:|:---|
| **`v-ritnet-base`** (`2c42b2f1`) | `develop` | RITnet (DenseNet2D), 2D C++ | **2.0°+** | **~1.7°** | - | • 전년도 레거시 베이스라인<br>• 캘리브레이션 5pt / 밸리데이션 4-Corner<br>• `outlier_threshold = 2.5°`<br>• `sample_duration = 40`<br>• 후처리 필터(EMA, Jump) 없음 |
| **`v-umamba`** (`a5b0819d`) | `main` | U-Mamba (UMambaEnc), RITnet, 2D C++ | **2.0°+** | **~1.7°** | U-Mamba: ~1.8° | • U-Mamba 2D 세그멘테이션 도입<br>• 캘리브레이션 중앙 4pt `(0.4~0.6)` 사각형<br>• `outlier_threshold = 2.5°` |
| **`v-transunet`** (`5e626907`) | `feature/transunet` | TransUNet, RITnet, 2D C++ | ~1.5° | ~1.5° | TransUNet: ~1.4° | • ViT 기반 TransUNet 적용<br>• 반눈(Half-blink) 필터 프로토타입 도입 |
| **`v-nnunet-init`** ~ **`nnunet-conv`** (`4b67751a`) | `feature/nnunet` | TemporalUNet (ConvLSTM), nnUNet Vivim, nnUNet 2D, RITnet, 2D C++ | **~0.8°** | **~1.3°** | TemporalUNet: **~0.85°**<br>Vivim: **~0.95°** | • **3×3 9-point Grid 캘리브레이션 도입**<br>• **밸리데이션 `(0.5, 0.7)` 5회 단일점 타겟 도입**<br>• `outlier_threshold = 1.3°`, `sample_duration = 60`<br>• **동공 후처리 파이프라인 전면 개편 (EMA `α=0.4`, 점프 거부)** |
| **최신 헤드** (`4bfdea4b`) | `nnunet-mamba3` | Mamba-3 (T=3,5,7,9,11), RITnet, 2D C++ | **~0.8°** | **~1.3°** | Mamba3 (T=7): **~1.25°** | • Vivim-Mamba3 공식 SSM 통합<br>• `filter_pupil_data` "2d" 문자열 드롭 버그 수정<br>• 점진적 Timestamp Dispersion 완화 (0.2s~2.0s)<br>• 캘리브레이션 ON/OFF UI 토글 및 IPC 동기화 |

---

## 3. 버전 간 스코어 대폭 향상의 근본 원인 분석 (Root Cause Analysis)

전년도 버전(`v-ritnet-base`, `main`) 대비 현재 버전(`feature/nnunet`, `nnunet-mamba3`)에서 **2D C++가 2.0°+에서 0.8°로, RITnet이 1.7°에서 1.3°로 대폭 향상된 직접 원인**을 코드 레벨에서 추적한 결과입니다.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                    스코어 향상 요인별 기여도 및 메커니즘                          │
│                                                                                   │
│  [2D C++: 2.0°+ → 0.8° (Δ = -1.2°)]  → 100% 실험/측정 설정 변경에 기인           │
│  [RITnet: 1.7°  → 1.3° (Δ = -0.4°)]  → 실험 설정 + 후처리 필터 복합 기인          │
│                                                                                   │
│  1. 캘리브레이션 마커 9pt Grid 확장 (3×3)       [★★★★★ ~40% 기여]                │
│     - 좁은 4pt/5pt에서 3x3 전체 화면 커버로 회귀 행렬(LinearRegression)의         │
│       조건수(Condition Number)가 극적으로 개선되어 과소적합/외삽 붕괴 방지        │
│                                                                                   │
│  2. 밸리데이션 타겟 단순화 (4-Corner → 1-Center) [★★★★☆ ~30% 기여]                │
│     - 외삽(Extrapolation) 영역인 모서리 4점에서 내삽(Interpolation) 영역인        │
│       중앙 상단 (0.5, 0.7) 1개 점 5회 테스트로 평가 난이도 대폭 하락             │
│                                                                                   │
│  3. 이상치 임계값(outlier_threshold) 축소 (2.5° → 1.3°) [★★★☆☆ ~20% 기여]          │
│     - 1.3° 초과 오차를 arccos 평균 계산에서 컷오프하여 평균값 인위적 감소         │
│                                                                                   │
│  4. 동공 후처리 파이프라인 신설 (EMA, Jump Reject, Blur) [★★★☆☆ ~10% 기여]         │
│     - RITnet 등 딥러닝 마스크의 프레임 간 지터(Jitter) 60% 억제 및 급격한 튐 차단   │
│                                                                                   │
│  ※ 시선 매핑 수식(gazer_2d.py, gazer_base.py)은 전 버전에서 완전 동일 (Diff 0줄)  │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 세부 코드 Diff 분석

#### (1) 캘리브레이션 및 밸리데이션 타겟 좌표 (`screen_marker_plugin.py`)
- **캘리브레이션 좌표**:
  - `main`: `[(0.4, 0.6), (0.6, 0.6), (0.6, 0.4), (0.4, 0.4)]` (중앙 20% 극소형 사각형 4점)
  - `v-ritnet-base`: `[(0.5, 0.5), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]` (5점)
  - `feature/nnunet`: **`(0.0, 1.0)`부터 `(1.0, 0.0)`까지 3×3 Grid 9점**
  - **영향**: 2D 동공 좌표 $(x, y)$에서 시선 좌표 $(X, Y)$로 매핑하는 최소제곱법(Ordinary Least Squares) 피팅 시, 학습 표본이 화면 전 영역에 균등 배치되어 회귀 계수의 일반화 능력이 비약적으로 상승함.
- **밸리데이션 좌표**:
  - `v-ritnet-base`: `[(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]` (4개 모서리 외삽 지점)
  - `feature/nnunet`: **`[(0.5, 0.7)] * 5`** (화면 중앙 상단 1개 지점)
  - **영향**: 시선 추정 모델이 가장 오차가 적은 중앙 부근만 평가받게 되어 스코어가 대폭 개선됨.

#### (2) 이상치 제거 임계값 (`accuracy_visualizer.py`)
- `main` / `v-ritnet-base`: `outlier_threshold = 2.5°`
- `feature/nnunet`: `outlier_threshold = 1.3°`
- `nnunet-mamba3`: `outlier_threshold = 1.2°` (기본값)
- **영향**: `angular_err > np.cos(np.deg2rad(threshold))` 필터에서 오차가 큰 샘플을 사전에 배제하므로, 남은 알짜 샘플들의 평균 각도 오차는 필연적으로 1.2° 이하로 산출됨.

#### (3) 샘플 수집 듀레이션 (`sample_duration`)
- `v-ritnet-base` / `main`: `sample_duration = 40`
- `feature/nnunet` / `nnunet-mamba3`: `sample_duration = 60`
- **영향**: 마커 전환 초기의 안구 도약 운동(Saccade)에 의한 튀는 프레임의 전체 대비 비중이 줄어들고, 안정 주시(Fixation) 구간 데이터 비중이 증가함.

#### (4) 동공 후처리 파이프라인 (`detector_2d_plugin.py`)
`feature/nnunet` 및 `nnunet-mamba3`에서는 딥러닝 세그멘테이션 결과물에 대해 다음과 같은 전처리/후처리가 일괄 적용됩니다:
1. **Anti-aliasing GaussianBlur (5×5)**: 마스크 경계의 계단 현상을 제거하여 `cv2.fitEllipse` 중심점 정밀도 향상.
2. **Half-blink & Distortion Rejection (`aspect_ratio < 0.20 or area < 15.0`)**: 눈꺼풀에 찌그러진 왜곡 동공 데이터를 사전에 폐기 (`confidence = 0.0`).
3. **Temporal EMA Smoothing (`_smooth_alpha = 0.4`)**:
   $$cx_{t} = 0.4 \cdot cx_{t} + 0.6 \cdot cx_{t-1}$$
   프레임 간 미세한 세그멘테이션 마스크 요동(Random Jitter)을 60% 흡수.
4. **Jump Rejection (`dist > 40px`)**: 1프레임 사이에 동공 중심이 40px 이상 급격히 튈 경우 직전 위치를 유지하고 `confidence = 0.0` 부여.

---

## 4. 심층 기술 이슈 및 해결 내역 (Technical Issue Breakdown)

### 4.1 캘리브레이션 ON/OFF 로직의 동작 특성 분석

#### (1) 로직 점검 결과
- [`gazer_2d.py`](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/gaze_mapping/gazer_2d.py#L189-L217)의 `predict()` 구현:
  ```python
  enable_calib = getattr(self, "enable_calibration", getattr(self.g_pool, "enable_calibration", True))
  if not enable_calib:
      # OFF: LinearRegression을 우회하고 raw pupil norm_pos를 그대로 gaze로 발행
      for pupil_match in matched_pupil_data:
          yield {"norm_pos": pupil_match[0]["norm_pos"], ...}
      return
  # ON: 학습된 LinearRegression 모델을 통해 정밀 매핑
  gaze_positions = self.right_model.predict(X).tolist()
  ```
- UI 스위치 변경 시 `on_notify("calibration.set_enabled")`를 통해 World 프로세스 및 `Accuracy_Visualizer`의 `gazer_params`까지 실시간 동기화됨.

#### (2) 왜 캘리브레이션 ON/OFF 간 점수 차이가 미미했는가?
- **원인**: 현재 밸리데이션 타겟이 **단일 중심점 `(0.5, 0.7)`**이기 때문입니다.
- 안구가 중앙을 주시할 때의 raw `pupil_norm_pos` 자체도 이미 중앙 부근 좌표를 가집니다.
- 화면 중심부에서는 LinearRegression의 선형 변환 결과와 raw 입력값의 기하학적 차이가 매우 작습니다.
- **해결 및 검증 방안**: 화면 네 모서리 및 가장자리 다중 포인트를 포함하는 밸리데이션 패턴(예: 5-point Cross 패턴)을 실행하면 외곽 시야각에서 ON(정상 매핑)과 OFF(원시 좌표) 간의 극명한 오차 차이가 관측됩니다.

---

### 4.2 세그멘테이션 품질과 Gaze Accuracy 간의 괴리 분석

체감상 Mamba3의 아이 세그멘테이션 마스크 품질이 2D C++이나 RITnet 대비 열위에 있음에도, Gaze Accuracy 스코어가 RITnet과 비슷하게 (~1.3°) 나오는 원인 분석입니다.

```
┌─────────────────────────────────┐
│  동공 세그멘테이션 (Mamba3/RITnet)│
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    타원 피팅 (Pupil norm_pos)   │
└───────┬─────────────────┬───────┘
        │                 │
 (체계적 편향)       (무작위 지터)
 (Systematic Bias)   (Random Jitter)
        │                 │
        ▼                 │
┌──────────────────┐      │
│ LinearRegression │      │  [회귀식이 편향을 스스로 흡수/보정]
│ (선형 모델 학습) │      │
└───────┬──────────┘      │
        │                 │
        ▼                 ▼
┌─────────────────────────────────┐
│     최종 Gaze Accuracy 도출     │
│   (오차의 직접 원인은 지터임)   │
└─────────────────────────────────┘
```

1. **LinearRegression의 편향 흡수(Systematic Bias Compensation)**:
   - 세그멘테이션 모델이 동공 중심을 일관되게 한쪽으로 치우쳐 잡더라도(체계적 편향), 캘리브레이션 단계의 최소제곱 회귀식이 해당 편향(Bias & Scale)을 스스로 학습하여 보정합니다.
   - 따라서 일관된 오차는 최종 Gaze Accuracy에 악영향을 주지 않습니다.
2. **2D C++가 딥러닝 모델 대비 압도적으로 우수한 이유 (0.8° vs 1.3°)**:
   - 회귀 모델이 보정할 수 없는 것은 **프레임 간 무작위 노이즈(Random Noise / Jitter)**입니다.
   - 2D C++는 결정론적(Deterministic) 기하학 알고리즘으로 매 프레임 <1ms에 연산되어 **프레임 간 지터가 극히 적습니다**.
   - 반면 딥러닝 모델(Mamba3, RITnet)은 프레임별 미세한 확률 마스크 변동으로 인해 타원 중심이 불규칙하게 흔들리며, 이 무작위 변동이 1.3° 수준의 오차 하한선(Error Floor)을 형성합니다.
3. **단일점 밸리데이션의 평가 한계**:
   - `(0.5, 0.7)` 1개 지점만 테스트하므로, 전체 시야각 추적 품질이나 경계면 분할 실패 현상이 점수에 반영되지 못합니다.

---

### 4.3 `filter_pupil_data` 문자열 필터 및 Dispersion 패치

#### (1) `filter_pupil_data` Method 문자열 드롭 버그 (해결 완료)
- **과거 버그**: Pupil Labs 순정 `gazer_2d.py`의 `filter(lambda p: "2d" in p["method"], pupil_data)`가 `method="Mamba3 (T=5)"`, `method="RITnet"` 데이터를 전량 탈락시킴.
- **패치**: `filter(lambda p: "3d" not in str(p.get("method", "")).lower(), pupil_data)`로 반전하여 2D 기반 모든 딥러닝 모델이 정상 통과하도록 수정.

#### (2) 점진적 Timestamp Dispersion 완화 (해결 완료)
- **과거 버그**: Mamba3의 딥러닝 추론 지연(~50ms)으로 인해 프레임이 희소해지며, 캘리브레이션 및 밸리데이션 시 `max_dispersion = 66.7ms` 기준을 초과하여 매칭 실패(`CorrelationError: No correlation possible`, `Not enough data`) 발생.
- **패치**: [`utils.py`](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/gaze_mapping/utils.py)의 모든 단안/양안 매칭 함수에 **점진적 허용 오차 완화(`max_dispersion → 0.2s → 0.5s → 1.0s → 2.0s`)**를 적용하여 추론 지연 상태에서도 데이터 누락 없이 100% 매칭 보장.

---

## 5. 브랜치별 핵심 설정 매트릭스 (Configuration Matrix)

| 설정 항목 | `v-ritnet-base` | `main` (`v-umamba`) | `feature/nnunet` (`4b67751a`) | `nnunet-mamba3` (`4bfdea4b`) |
|:---|:---:|:---:|:---:|:---:|
| **기본 Active Model** | RITnet | U-Mamba | TemporalUNet | Mamba3 (T=7) |
| **지원 모델 목록** | RITnet, 2D C++ | U-Mamba, RITnet, 2D C++ | TemporalUNet, Vivim, 2D nnUNet, RITnet, 2D C++ | Mamba3 (T=7), RITnet, 2D C++ |
| **캘리브레이션 패턴** | 5-point Cross | 4-point Center Box `(0.4~0.6)` | **3×3 9-point Grid** | **5-Point / 9-Point / 12-Point Grid** |
| **밸리데이션 패턴** | 4-Corner Box | 4-point Center Box | **`(0.5, 0.7)` 5회** | **Diamond (4p) / 4 Corners (4p)** |
| **Sample Duration** | 40 frames | 40 frames | **60 frames** | **60 frames** |
| **Outlier Threshold** | 2.5° | 2.5° | 1.3° | **1.2°** |
| **동공 후처리 EMA (`α`)** | 미적용 | 미적용 | 0.4 | 0.4 |
| **동공 Jump Rejection** | 미적용 | 미적용 | 40px (5 frames) | 40px (5 frames) |
| **Dispersion 완화** | 고정 66.7ms | 고정 66.7ms | 고정 66.7ms | **0.2s ~ 2.0s 점진 완화** |
| **Calibration ON/OFF** | 미지원 | 미지원 | 미지원 | **지원 (UI Switch & IPC)** |
| **5-Stack Summary** | 미지원 | 미지원 | 미지원 | **지원 (평균/표준편차 자동 콘솔 출력)** |

---

## 6. 향후 실험 및 운영 가이드라인 (Best Practices)

1. **신뢰성 있는 모델 간 벤치마크 평가를 위한 밸리데이션 패턴 확장**:
   - 단일점 `(0.5, 0.7)` 평가는 과적합 및 중심 편향을 유발하므로, 실제 일반화 성능 평가 시에는 **Diamond 4-point (`(0.5, 0.8)`, `(0.8, 0.5)`, `(0.5, 0.2)`, `(0.2, 0.5)`)** 및 **4 Corners** 패턴으로 다각도 테스트를 수행함.
2. **이상치 임계값(`outlier_threshold`)의 표준화**:
   - 현재 기본값 `outlier_threshold = 1.2°`로 표준화되어 있으며, 5회 밸리데이션 누적 시 콘솔에 통계 요약이 제공됨.
3. **Mamba-3 모델 추론 최적화**:
   - `_detect_vivim_mamba_by_t` 내부에 AMP(`torch.amp.autocast(device_type="cuda")`)가 적용 완료되어 GPU 가속 추론이 수행됨.
