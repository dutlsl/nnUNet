# 🛠️ Vivim Backbone Mamba Module & WeakMed Loss Technical Specification

> **문서 개요**: 현재 커밋(`weakmed-v2` 브랜치) 기준 Vivim 백본 아키텍처 내 Mamba SSM 모듈 탑재 상태 및 WeakMed (CVPR 2025) 약한 지도 학습 손실 함수 구현 명세서입니다.

---

## 📌 1. Vivim (Video Vision Mamba) 백본 구현 명세

### 1) 백본 구조 (`models/vivim_backbone.py`)
* **입력 텐서 규격**: `[B, T=3, C=1, H=448, W=448]` (3-frame temporal sequence)
* **엔진 분기**:
  - 2D PlainConvUNet 인코더 ($E_1 \rightarrow E_2 \rightarrow E_3$)를 프레임 단위로 병렬 통과 ($[B \cdot T, C, H, W]$)
  - 병목(Bottleneck) 구간에서 시계열 피처 차원($[B, T, 256, H/8, W/8]$)으로 재구성 후 **`TemporalMambaBlock`** 실행
  - 2D 디코더 단에서 마지막 프레임($T$) 병목 피처 및 Skip Connection 결합 후 세그멘테이션 출력

### 2) Mamba SSM 모듈 (`models/mamba_block.py`, `models/temporal_mamba.py`)
* **Selective Scan Engine**:
  - `mamba_ssm.ops.selective_scan_interface` (CUDA 커스텀 커널) 자동 감지
  - CUDA 환경 불가능 시 Pure PyTorch Autograd 기반 `selective_scan_pytorch` 수식 엔진 자동 분기
* **파라미터 설정**:
  - `d_model`: 256 (병목 채널 수)
  - `d_state`: 16 (SSM 내부 상태 차원)
  - `d_conv`: 4 (Sequence Conv1d 커널 크기)
  - `expand`: 2 (채널 확장 비율)

---

## 📌 2. WeakMed Loss 함수 구현 명세 (`losses/weakmed_loss.py`)

CVPR 2025 WeakMed 논문 원문 수식 기반의 약한 지도학습/반지도학습 손실 함수가 구현되어 있습니다.

### 1) Mask-to-Box (M2B) Transform Loss (`M2BLoss`)
* **원문 수식 매핑 (Eq. 1, 2, 3)**:
  1. **Projection (Eq. 1)**: 전경 영역 패치 $P'$에 대해 행/열 맥스 풀링 $P_w = \max(P', \text{dim}=0)$, $P_h = \max(P', \text{dim}=1)$ 계산
  2. **Back-projection (Eq. 2)**: $T' = \min(P_w, P_h)$ 외적 최소값 연산으로 박스 정렬 변환 마스크 복원 후 전역 마스크 $T$ 합성
  3. **Box Supervision (Eq. 3)**: 전체 이미지 공간 상에서 바운딩 박스 정답 $B$와의 복합 손실 계산:
     $$\mathcal{L}_{\text{M2B}} = 0.5 \times \mathcal{L}_{\text{BCE}}(T, B) + 0.5 \times \mathcal{L}_{\text{Dice}}(T, B)$$

### 2) Sclera Containment Loss (`ScleraContainmentLoss`)
* **반지도 제약 수식**:
  - 2D 세그멘테이션 헤드에서 예측된 공막(Sclera) 픽셀이 `SphereHead`가 추정한 안구 구형 영역 내에 100% 포함되도록 강제하는 제약 손실
  $$\mathcal{L}_{\text{containment}} = \text{mean}\left(\max(0, M_{\text{sclera}} - M_{\text{sphere}})^2\right)$$

### 3) 퓨전 복합 손실 클래스
* **`WeakMedSphereLoss`**: v2 트레이너 연동 손실 ($\mathcal{L}_{\text{DiceCE}} + \mathcal{L}_{\text{M2B}} + 0.5 \times \mathcal{L}_{\text{containment}}$)
* **`WeakMedEyeballLoss`**: v1 트레이너 연동 손실 ($\mathcal{L}_{\text{M2B}} + \mathcal{L}_{\text{SC}}$)

---

## 📌 3. 트레이너 연동 현황 (`nnUNetTrainer_Vivim.py`)

* **프레임워크**: `nnUNet v2` 네이티브 커스텀 트레이너
* **실행 엔진**: `train_step` 및 `validation_step` 내에서 `VivimBackbone` 5D 시퀀스 전방 전달 및 `WeakMedSphereLoss` 역전파 구동
* **학습 스케줄**: 1,000 에폭, PolyLRScheduler, AdamW 옵티마이저, WandB 대시보드 자동 동기화

---

## 📌 4. WeakMed v2 RITnet 전처리 모사 & 192×192 해상도 명세

Pupil Labs 시연 환경(RITnet / Pye3D 연동)과의 도메인 갭을 최소화하기 위해 WeakMed v2 학습 파이프라인에 **RITnet 전처리 모사 파이프라인** 및 **192×192 인풋 규격**을 적용하였습니다.

### 1) RITnet 전처리 모사 파이프라인 (`get_img` 모사)
* **Gamma Correction ($\gamma=0.8$)**:
  - `table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)`
  - `cv2.LUT(gray_img, table.astype(np.uint8))`를 적용하여 조명이 어두운 IR 안구 영상의 톤 커브 보정
* **CLAHE (Contrast Limited Adaptive Histogram Equalization)**:
  - `cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))`
  - 국소 영역 명암 대비를 향상시켜 동공(Pupil)과 홍채(Iris) 간 픽셀 경계부 감지력 극대화
* **Image Normalization**:
  - `torchvision.transforms.Normalize([0.5], [0.5])`
  - 픽셀 값 수렴 범위를 기존 Z-score 대신 `[-1.0, 1.0]` 범위로 정규화

### 2) 입력 텐서 해상도 규격 (192×192 vs 400×640)
* **WeakMed v2 192×192 학습**: RITnet 및 C++ 2D Detector 모듈 입력 호환을 위해 `192 × 192` 크기로 리사이즈/패딩된 시퀀스 피처맵 사용
* **영향**: 시연 서버의 RITnet 2D 타원 피팅 헤드와의 피처 스케일 이질성을 해소하고 inference 연산 속도를 상향 유도함

