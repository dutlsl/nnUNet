# Pupil 2D 동공 검출기 딥러닝 모델 확장 및 연동 가이드

---

## 2. 기존 모델별 전·후처리 및 호출 구조 분석

### 2.1 Vivim-Mamba3 (T=7)
- **위치**: [`pupil_src/shared_modules/pupil_detector_plugins/vivim/`](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/pupil_detector_plugins/vivim/)
- **가중치**: `pupil_src/shared_modules/pupil_detector_plugins/best_checkpoint_t7.pth`
- **학습 도메인 파이프라인**:
  - OpenEDS 데이터셋 규격: 동적 Z-Score 정규화 (`(gray - mean) / (std + 1e-8)`)
  - 400×400 리사이즈 후 448×448 제로 패딩 (24px 경계)
  - 7개 프레임 시퀀스 슬라이딩 윈도우: `[1, 7, 1, 448, 448]`
- **추론**: `torch.inference_mode()` + `torch.amp.autocast(device_type="cuda")`
- **출력**: `[1, 4, 448, 448]` Logits → `argmax(dim=1)` → `[24:424, 24:424]` 24px Unpad → 원본 해상도 매핑

### 2.2 RITnet (DenseNet2D)
- **위치**: [`pupil_src/shared_modules/pupil_detector_plugins/densenet.py`](file:///home/byeongjun/PycharmProjects/pupil/pupil_src/shared_modules/pupil_detector_plugins/densenet.py)
- **가중치**: `pupil_src/best_model.pkl`
- **입력 전처리**: Gamma 0.8 + CLAHE 1.5 + `Normalize([0.5], [0.5])` (`[-1.0, 1.0]`)
- **출력**: `[1, 4, H, W]` Logits → `get_predictions()` → `[H, W]` Mask

### 2.3 2D C++ (Pupil Labs 기본 검출기)
- **위치**: `pupil_detectors` C++ 바이너리 라이브러리
- **특징**: 전통 컴퓨터 비전 에지 기반 타원 피팅 알고리즘

---

## 3. 신규 딥러닝 모델 추가 5단계 체크리스트

새로운 모델(예: `MyNewModel`)을 추가하여 실시간 추론 및 정확도 평가를 진행하고자 할 때 아래 순서대로 수정합니다.

### 1단계: 모델 코드 및 체크포인트 배치
- 모델 정의 파일 생성: `pupil_src/shared_modules/pupil_detector_plugins/mynewmodel/model.py`
- 학습된 체크포인트 파일 배치: `pupil_src/shared_modules/pupil_detector_plugins/best_mynewmodel.pth`

### 2단계: `detector_2d_plugin.py`에 모델 임포트 및 초기화 메소드 작성
```python
# pupil_detector_plugins/detector_2d_plugin.py

# 1. 모델 클래스 임포트
from .mynewmodel import MyNewModel

class Detector2DPlugin(PupilDetectorPlugin):
    def __init__(self, ...):
        ...
        self.mynew_model = None
        self._init_mynew_model()

    def _init_mynew_model(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ckpt_path = os.path.join(current_dir, "best_mynewmodel.pth")
            if os.path.exists(ckpt_path):
                self.mynew_model = MyNewModel(...).to(self.device)
                self.mynew_model.load_state_dict(
                    torch.load(ckpt_path, map_location=self.device, weights_only=False)
                )
                self.mynew_model.eval()
                logger.info("✅ MyNewModel initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize MyNewModel: {e}")
```

### 3단계: UI 드롭다운 메뉴 항목 추가
`init_ui()`의 `active_model` Selector에 신규 모델 명칭을 추가합니다.
```python
def init_ui(self):
    super().init_ui()
    ...
    self.menu.append(
        ui.Selector(
            "active_model",
            self,
            label="Active Model",
            selection=["Mamba3 (T=7)", "MyNewModel", "RITnet", "2D C++"],
        )
    )
```

### 4단계: `detect()` 라우팅 및 추론 함수 구현
```python
def detect(self, frame, **kwargs):
    active = getattr(self, "active_model", "Mamba3 (T=7)")
    ...
    elif active == "MyNewModel":
        if self.mynew_model is None:
            return self._empty_datum(frame)
        return self._detect_mynew_model(frame, **kwargs)

def _detect_mynew_model(self, frame, **kwargs):
    gray = frame.gray
    if gray is None or self.mynew_model is None:
        return self._empty_datum(frame)

    gray = gray.astype(np.uint8)
    orig_h, orig_w = gray.shape[:2]
    
    # 1. Flip 처리
    flip_v = getattr(self, "flip_vertically", False)
    flip_h = getattr(self, "flip_horizontally", False)
    if flip_v: gray = cv2.flip(gray, 0)
    if flip_h: gray = cv2.flip(gray, 1)

    # 2. 모델 학습 스펙에 맞는 전처리 적용
    img_tensor = self.get_img(gray)
    data = img_tensor.unsqueeze(0).to(self.device)

    # 3. AMP 추론
    with torch.inference_mode():
        if self._use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = self.mynew_model(data)
        else:
            logits = self.mynew_model(data)

    pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # 4. 공통 후처리 파이프라인 호출
    return self._postprocess_mask_to_datum(pred_mask, frame, orig_h, orig_w, flip_v, flip_h)
```

### 5단계: 후처리 규약 (`_postprocess_mask_to_datum`)
`_postprocess_mask_to_datum()`은 세그멘테이션 마스크로부터 동공 클래스(`PUPIL_CLASS_ID = 3`)를 추출하여:
- 가우시안 블러 및 이진화 (Anti-aliasing)
- 최대 컨투어 검색 및 타원 피팅 (`cv2.fitEllipse`)
- 블링크 및 왜곡 동공 필터 (`aspect_ratio < 0.20 or area < 15.0`)
- 지수 이동 평균(EMA) 스무딩 (`α = 0.4`) 및 점프 거부 (`40px`)
- 정규화 좌표(`norm_pos`) 및 타원 기하 정보를 담은 Pupil `datum` 딕셔너리 생성을 일괄 처리합니다.
