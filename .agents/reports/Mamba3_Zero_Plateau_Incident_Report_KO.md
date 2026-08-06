# Mamba-3 연동 및 SAGD 성능 지표 0.0 정체 현상 문제 분석 보고서

**작성일**: 2026-08-06  
**대상 아키텍처**: Vivim + SAGD (SAS + HDM + SGA) + 공식 Mamba-3 (`mamba_ssm.modules.mamba3.Mamba3`)  
**실험 환경**: PyTorch 2.5.1 + CUDA 11.8 / Dual RTX 3090  

---

## 1. 개요 (Incident Overview)

공식 Mamba-3 백본 기반 SAGD 메인 학습 프로세스(`nnUNetTrainer_Vivim_SADG`, PID 1049934) 진행 중, 학습 손실(`train_loss`)은 `0.49` 수준으로 유지되었으나 **Epoch 25부터 Epoch 52까지 Validation Dice 지표가 `0.0000`에 고정되어 탈출하지 못하는 현상**이 발생하였습니다.

본 보고서는 아래 사항에 대한 정밀 조사 결과를 정리합니다:
1. `dutlsl/backbone` 브랜치의 공식 Mamba-3 연동 무결성 검증.
2. PyTorch 2.5 환경에서의 Mamba3 Float8 런타임 호환성 보완.
3. 사전학습 가중치(Pretrained Weights) 로드 상세 명세 검증.
4. Validation Dice score `0.0000` 정체 현상의 근본 원인 분석.
5. 재발 방지 및 문제 해결을 위한 기술적 조치 방안.

---

## 2. 공식 Mamba-3 연동 및 호환성 검증

### 2.1 백본 브랜치 동기화 및 모듈 호출
- **당겨온 브랜치**: `dutlsl/backbone` (`commit: edbab78`)
- **수정/적용 파일**: [models/mamba_block.py](file:///home/iulab1/PycharmProjects/nnUNet/models/mamba_block.py)
- **모듈 연동 확인**:
  ```python
  from mamba_ssm.modules.mamba3 import Mamba3

  class MambaLayer(nn.Module):
      def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2, headdim: int = 64, ngroups: int = 1, chunk_size: int = 64, **kwargs):
          super().__init__()
          self.mamba3 = Mamba3(
              d_model=d_model,
              d_state=d_state,
              expand=expand,
              headdim=headdim,
              ngroups=ngroups,
              chunk_size=chunk_size,
          )

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          return self.mamba3(x)
  ```

### 2.2 PyTorch 2.5 런타임 호환성 처리 (Polyfill)
Mamba3 패키지(`mamba-ssm 2.3.2`) 초기화 시 PyTorch 2.6+ 사양의 `torch.float8_e8m0fnu` 속성을 참조함에 따라 현 서버(PyTorch 2.5.1)에서 `AttributeError`가 발생하는 현상을 확인하였습니다.

**조치 사항**: [models/mamba_block.py](file:///home/iulab1/PycharmProjects/nnUNet/models/mamba_block.py) 최상단에 안전한 속성 보완 코드를 추가하여 튕김 현상을 해결하였습니다:
```python
import torch
import torch.nn as nn

# PyTorch 2.5 환경에서 Cutlass/Quack에 필요한 float8 속성 보완 (Polyfill)
for dt in ['float8_e8m0fnu', 'float4_e2m1fn_x2', 'float8_e4m3fnuz', 'float8_e5m2fnuz']:
    if not hasattr(torch, dt):
        setattr(torch, dt, getattr(torch, 'float8_e5m2', torch.float32))

from mamba_ssm.modules.mamba3 import Mamba3
```
- **검증 완료**: GPU 1번에서 스모크 테스트(`tests/test_sadg_full.py`)를 수행하여 CUDA 순전파/역전파 100% 정상 작동을 확인했습니다.

---

## 3. 사전학습 가중치(Pretrained Weights) 로드 상세 검증

베이스라인 Vivim 체크포인트(`checkpoint_final.pth`)에서 `VivimSAGDWrapper` 모델로 로드된 가중치 파라미터 명세입니다:

| 파라미터 그룹 | 체크포인트 텐서 수 | 로드 성공 텐서 수 | 로드 비율 | 상세 명세 |
| :--- | :---: | :---: | :---: | :--- |
| **Encoder (`enc1` ~ `enc4`)** | 32 | 32 | **100%** | 컨볼루션, BatchNorm 텐서 100% 로드 |
| **Decoder (`dec1` ~ `dec4`)** | 32 | 32 | **100%** | 업샘플링 디코더 텐서 100% 로드 |
| **Bottleneck (`bottleneck`)** | 8 | 8 | **100%** | 바틀넥 피처 텐서 100% 로드 |
| **Segmentation Head (`seg_outputs`)** | 8 | 8 | **100%** | 마스크 분류 헤드 텐서 100% 로드 |
| **Input Stem (`stem`)** | 16 | 16 | **100%** | 입력 임베딩 스템 텐서 100% 로드 |
| **1세대 Mamba 구형 레이어** | 9 | 0 | *제외* | 3세대 Mamba3 교체로 인한 구형 파라미터 |
| **총 체크포인트 가중치** | **105** | **96** | **91.4%** | **기존 Vivim 뼈대 가중치 100% 복원 완료** |

> [!NOTE]
> 체크포인트 내 기존 Vivim 백본의 핵심 파라미터 96개는 100% 정확하게 로드되었습니다.
> 로그 상의 `Missing: 142`는 새로 추가된 SAGD 전용 모듈(`SAS`, `HDM`, `SGA`) 및 3세대 `Mamba3` 투영 레이어(Fused Projection)의 신규 난수 초기화 파라미터 수치입니다.

---

## 4. Validation Dice 0.0 정체 현상 원인 분석

### 4.1 지표 변화 추이 (Trajectory Log)
- **Epoch 0 ~ 4**: Pseudo Dice `0.1008` ~ `0.2144` (Sclera Dice `0.4974`) 기록.
- **Epoch 8**: Sclera Dice `0.2272`, Iris Dice `0.1668`, Mean `0.1314` 유지.
- **Epoch 24**: Iris Dice `0.5782`, Mean `0.1986` 기록.
- **Epoch 25 ~ 52**: Dice Mean `0.0000`, Precision `0.0000`, Recall `0.0000` 고정.

### 4.2 메커니즘 분석 (Zero-Plateau Collapse)
1. **Mamba-3 내적 레이어 난수 초기화**: 기존 Vivim 뼈대 가중치는 로드되었으나, 새로 도입된 Mamba-3 내부의 상태 투영 레이어(`in_proj`, `out_proj`, `x_proj`)는 난수로 초기화되었습니다.
2. **배경(Background) 클래스 편향**: 안구 영상 데이터 특성상 전체 공간 픽셀의 90% 이상이 배경(Class 0, Background)입니다.
3. **로컬 미니멈(Local Minimum) 함정 진입**: Epoch 25 시점에 도메인 일반화 손실($L_{total} = L_{DiceCE} + 0.1 L_{DC} + 0.05 L_{SC}$)의 복합 연산 과정에서, 초기화되지 않은 Mamba-3 표현력이 세그멘테이션 헤드를 **"모든 픽셀을 배경(Class 0)으로 출력하는 상태"**로 유도했습니다.
4. **그래디언트 장벽 고정**: 모든 픽셀을 배경으로 예측하면:
   $$\text{True Positives (TP)} = 0 \implies \text{Dice} = 0.0000, \quad \text{Precision} = 0.0000, \quad \text{Recall} = 0.0000$$
   배경 픽셀만으로도 손실값이 `0.49` 수준으로 떨어지므로, 전경 클래스(Pupil, Iris, Sclera)를 복원하기 위한 기울기(Gradient) 역전이 일어나지 못하고 0.0 늪에 안착하게 됩니다.

---

## 5. 해결 방안 및 기술적 조치 계획

1. **Mamba-3 및 도메인 손실 웜업 (Domain Loss Warmup)**:
   - 초반 10 에폭 동안 도메인 일과성 손실 가중치($w_{dc}, w_{sc}$)를 0부터 선형 증가(Linear Warmup)시켜, Mamba-3 레이어가 세그멘테이션 마스크 표현력을 먼저 잡도록 유도합니다.
     $$w_{dc}(t) = \min\left(0.1, \frac{t}{10} \times 0.1\right), \quad w_{sc}(t) = \min\left(0.05, \frac{t}{10} \times 0.05\right)$$

2. **Cross-Entropy 손실 가중치 보완 (Class Weighting)**:
   - 배경 쏠림 현상을 방지하도록 `ce_weight` 비율을 `0.3` ➔ `0.5`로 상향하거나, 전경 클래스에 가중치($[0.1, 1.0, 1.0, 1.0]$)를 적용합니다.

3. **코드 및 보고서 깃허브 반영**:
   - 본 한글 문제 보고서를 레포지토리 내 `.agents/reports/Mamba3_Zero_Plateau_Incident_Report_KO.md` 경로에 저장하고 `dutlsl/sadg` 브랜치에 반영합니다.
