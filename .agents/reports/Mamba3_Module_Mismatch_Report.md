# 🚨 Mamba 모듈 탑재 불일치 분석 보고서 (Mamba-3 & mamba_ssm 연동 문제)

> **문서 작성 일시**: 2026-08-06  
> **보고 목적**: `models/mamba_block.py` 및 `models/temporal_mamba.py` 내 Mamba 모듈 연동 방식의 불일치 및 Mamba-3 (arXiv:2603.15569) 공식 릴리즈 패키지 호출 누락 문제 분석 보고  

---

## 📌 1. 문제 상황 요약

1. **공식 패키지/모듈 호출 누락**:
   - `mamba_ssm` 패키지(v2.2.4)가 가상환경에 설치되어 있었으나, `models/mamba_block.py` 코드가 패키지의 공식 모듈 인스턴스(`from mamba_ssm import Mamba` 또는 `Mamba2`)를 직접 가져와 사용하지 않고, 1세대 `selective_scan_fn` c++ 커널 인터페이스와 커스텀 프로젝션 층(`in_proj`, `x_proj`, `dt_proj`)을 수동으로 재구현하여 연동하고 있었음.
   
2. **Mamba-3 (arXiv:2603.15569) 최신 스펙 미반영 문제**:
   - 2026년 3월 릴리즈된 **Mamba-3 (arXiv:2603.15569)** 공식 저장소(`state-spaces/mamba`)의 최신 수식(Exponential-Trapezoidal Discretization, Complex-Valued State Update, MIMO) 및 최신 API 모듈이 코드베이스에 올바르게 적용되지 않고 기존 1세대 Selective Scan 수동 파이프라인이 유지되었음.

---

## 📌 2. 상세 코드 분석 (`models/mamba_block.py`)

### 현재 코드 상태
```python
# models/mamba_block.py
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        # 수동 프로젝션 층 정의
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        ...

    def forward(self, x: torch.Tensor):
        ...
        # 1세대 C++ 커널 직접 연동
        y = selective_scan_fn(x_act, dt, A, B_param, C_param, self.D)
        ...
```

### 지적된 구조적 결함
- `mamba_ssm` 패키지(`state-spaces/mamba`)에서 제공하는 공식 릴리즈 클래스를 직접 인스턴스화하지 않고 커스텀 레이어로 감싸면서 최신 Mamba-3 패키지 기능 및 최적화가 우회됨.

---

## 📌 3. 해결 방안 (Action Plan)

1. **공식 Mamba3 / Mamba 패키지 직접 연동**:
   - `models/mamba_block.py` 내부에서 수동 C++ 커널 바인딩 방식을 제거하고, `state-spaces/mamba` 공식 릴리즈 모듈을 직접 호출하도록 리팩토링.
2. **`mamba_ssm` 최신 릴리즈 커널 환경 동기화**:
   - Mamba-3 (arXiv:2603.15569) 사양에 맞는 패키지 모듈 호출 방식으로 `TemporalMambaBlock` 재구성.

---

*본 보고서는 `backbone` 브랜치에 기록되어 향후 백본 리팩토링 시 참고자료로 활용됩니다.*
