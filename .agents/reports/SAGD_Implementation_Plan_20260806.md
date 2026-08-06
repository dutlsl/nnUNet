# SAGD 실험 실패 원인 원천 차단 + 코드 리팩토링

## 실패 문서 분석 결과 요약

6건의 실패 문서를 전수 분석하여 **코드에 남아있는 버그와 구조적 결함**을 분류했습니다.

| 문서 | 핵심 발견 |
|------|-----------|
| [SAGD_NaN_Bug_Report](file:///home/iulab1/PycharmProjects/nnUNet/.agents/reports/SAGD_NaN_Bug_Report_20260806.md) | **🔴 AMP FP16에서 KL Divergence log(0) = -inf, InfoNCE /0.07 오버플로우 → Epoch 201 NaN 붕괴** |
| [SAGD_NaN_Bug_Report](file:///home/iulab1/PycharmProjects/nnUNet/.agents/reports/SAGD_NaN_Bug_Report_20260806.md) | **🔴 `nan_to_num(l, nan=0.5)` 로 NaN 은폐 → 오염된 gradient가 전체 가중치 파괴** |
| [SAGD_Loss_Design_Analysis](file:///home/iulab1/PycharmProjects/nnUNet/.agents/reports/SAGD_Loss_Design_Analysis_20260806.md) | Train/Val loss 수식 불일치는 의도된 설계 (수정 불필요) |
| [v3_architectural_problem_analysis](file:///home/iulab1/PycharmProjects/nnUNet/.agents/reports/v3_architectural_problem_analysis_report.md) | WeakMed SphereHead 관련 → 현 SAGD 브랜치에서 이미 제거됨 |
| [WeakMEd_Problem_Report](file:///home/iulab1/PycharmProjects/nnUNet/.agents/reports/WeakMEd_Problem_Report.md) | Eyeball head box collapse → 현 SAGD 브랜치에서 이미 제거됨 |
| [SKILL.md Lesson Learned](file:///home/iulab1/PycharmProjects/nnUNet/.agents/skills/research-code-harness/SKILL.md) | 하드코딩 경로, 모듈 경계 부재, 재현성 부재 → 부가 리팩토링 |

---

## 🔴 핵심 버그 1: AMP FP16 NaN 오버플로우/언더플로우 (원천 차단)

### 근본 원인

**로그 증거**: Epoch 200 `train_loss: 0.5869, val_loss: 1.1348` → Epoch 201 `val_loss: nan` → Epoch 202~ `train_loss: 0.5000 고정`

두 곳에서 FP16 정밀도 한계를 초과하는 연산이 발생:

#### 1-A. `DomainConsistencyLoss`: `torch.log(p_soft + 1e-8)` 언더플로우

[sadg_loss.py:127-132](file:///home/iulab1/PycharmProjects/nnUNet/losses/sadg_loss.py#L127-L132)

```python
# 현재 코드 (위험)
kl_pa = F.kl_div(
    torch.log(p_soft + 1e-8), a_soft, reduction='none'  # ❌ FP16에서 1e-8은 0으로 잘림 → log(0) = -inf
).sum(dim=1).mean()
```

- FP16 최소 표현 가능 수치: ~6×10⁻⁵
- `p_soft`가 0에 가까울 때 `1e-8` 더해봤자 FP16에서 0으로 flush → `log(0) = -inf`

#### 1-B. `StructuralContrastiveLoss`: `/ self.temperature` (0.07) 오버플로우

[sadg_loss.py:185](file:///home/iulab1/PycharmProjects/nnUNet/losses/sadg_loss.py#L185)

```python
# 현재 코드 (위험)
sim_matrix = torch.bmm(anchors, a_norm.transpose(1, 2)) / self.temperature  # ❌ /0.07 → FP16 max(65504) 초과
sim_matrix = torch.clamp(sim_matrix, min=-50.0, max=50.0)  # ❌ clamp 전에 이미 +inf 발생
```

- 내적 값이 FP16에서 `/0.07` ≈ `×14.3` 증폭 시 `65504` 초과 → `+inf` → `NaN` 전파

### 수정 방안

```python
# 1-A: KL Divergence - FP32로 강제 캐스팅 + 안전 clamp
p_soft = F.softmax((primary_logits / self.temperature).float(), dim=1)
a_soft = F.softmax((aux_logits / self.temperature).float(), dim=1)
kl_pa = F.kl_div(
    torch.log(p_soft.clamp(min=1e-6)), a_soft, reduction='none'
).sum(dim=1).mean()

# 1-B: InfoNCE - FP32 강제 캐스팅 + temperature 나누기 전 clamp
anchors = p_norm[:, indices, :].float()
a_norm_f = a_norm.float()
sim_matrix = torch.bmm(anchors, a_norm_f.transpose(1, 2))
sim_matrix = torch.clamp(sim_matrix, min=-1.0, max=1.0)  # 코사인 유사도 범위 제한
sim_matrix = sim_matrix / self.temperature  # clamp 후 나누기 → 오버플로우 불가
```

---

## 🔴 핵심 버그 2: `nan_to_num(l, nan=0.5)` NaN 은폐 → 가중치 파괴 (원천 차단)

### 근본 원인

[nnUNetTrainer_Vivim_SADG.py:508-509](file:///home/iulab1/PycharmProjects/nnUNet/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainer_Vivim_SADG.py#L508-L509)

```python
# 현재 코드 (치명적)
l = loss_dict['total']
if torch.isnan(l) or torch.isinf(l):
    l = torch.nan_to_num(l, nan=0.5, posinf=1.0, neginf=0.0)  # ❌ 가짜 loss로 역전파 강제 진행
```

- NaN loss가 발생하면 **해당 step을 건너뛰어야** 하는데, 가짜 값 0.5를 부여하고 `backward()` 진행
- 오염된 gradient가 누적 → 전체 네트워크 가중치 붕괴

### 수정 방안

```python
l = loss_dict['total']
if torch.isnan(l) or torch.isinf(l):
    # 안전한 step skip: gradient 초기화 후 해당 step 건너뛰기
    self.optimizer.zero_grad(set_to_none=True)
    print(f"[SAGD WARNING] NaN/Inf loss detected, skipping step", flush=True)
    return {'loss': 0.0}  # 로깅용 0.0 반환, backward() 실행 안 함
```

---

## 🟡 구조적 결함: validation_step에서 DiceCELoss 매번 재생성

[nnUNetTrainer_Vivim_SADG.py:540-542](file:///home/iulab1/PycharmProjects/nnUNet/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainer_Vivim_SADG.py#L540-L542)

```python
# 현재 코드 (비효율)
from losses.sadg_loss import DiceCELoss  # ❌ 매 step마다 import
val_loss_fn = DiceCELoss(num_classes=seg_logits.shape[1])  # ❌ 매 step마다 인스턴스 생성
```

### 수정 방안: `_build_loss()`에서 한 번만 생성하여 `self.val_loss_fn`에 캐시

---

## 🟢 부가 리팩토링: 하드코딩 경로 제거 (서버 이식성)

| 대상 | 하드코딩 | 수정 |
|------|----------|------|
| `sadg_vivim.yaml` | `/home/iulab1/...` 절대 경로 5곳 | 환경변수 `NNUNET_PROJECT_ROOT` 기반 상대 경로 |
| `multi_domain_dataset.py` | LPW PNG 경로 4곳 | config에서 `extracted_video_root`, `extracted_label_root` 주입 |
| `nnUNetTrainer_Vivim_SADG.py:393` | `/home/iulab0/...` pretrained 경로 | config `training.pretrained_checkpoint` 경로 |
| `wandb.init` | `mode` 미지정 | 환경변수 `WANDB_MODE` 자동 반영 |

---

## Proposed Changes (수정 순서)

### [MODIFY] [sadg_loss.py](file:///home/iulab1/PycharmProjects/nnUNet/losses/sadg_loss.py) — 🔴 NaN 원천 차단

1. `DomainConsistencyLoss.forward()`: `.float()` FP32 강제 + `clamp(min=1e-6)` 적용
2. `StructuralContrastiveLoss.forward()`: `.float()` FP32 강제 + `clamp(-1, 1)` 후 temperature 나누기 순서 변경

---

### [MODIFY] [nnUNetTrainer_Vivim_SADG.py](file:///home/iulab1/PycharmProjects/nnUNet/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainer_Vivim_SADG.py) — 🔴 NaN 은폐 제거 + 🟡 val_loss_fn 캐시 + 🟢 하드코딩 제거

1. `train_step()`: `nan_to_num` → 안전한 step skip으로 교체
2. `_build_loss()`: `self.val_loss_fn` 추가 생성
3. `validation_step()`: 인라인 DiceCELoss 생성 → `self.val_loss_fn` 사용
4. `initialize_network()`: 하드코딩 pretrained 경로 → config 기반
5. `on_train_start()`: wandb mode 환경변수 반영 (offline 지원)
6. Config 로딩: 동적 프로젝트 루트 해석

---

### [MODIFY] [sadg_vivim.yaml](file:///home/iulab1/PycharmProjects/nnUNet/configs/sadg_vivim.yaml) — 🟢 경로 상대화

1. 모든 절대 경로 → 환경변수 기반 상대 경로
2. `training.pretrained_checkpoint` 추가
3. `data.lpw.extracted_video_root`, `extracted_label_root` 추가

---

### [MODIFY] [multi_domain_dataset.py](file:///home/iulab1/PycharmProjects/nnUNet/datasets/multi_domain_dataset.py) — 🟢 하드코딩 제거

1. `LPWDomainDataset.__init__()`: `extracted_video_root`, `extracted_label_root` 파라미터 추가
2. `_read_video_frames()`: 하드코딩 경로 → 인스턴스 변수 참조
3. `_read_label_frame()`: 하드코딩 경로 → 인스턴스 변수 참조
4. `get_multi_domain_dataloaders()`: config에서 새 경로 전달

---

### [MODIFY] [config.py](file:///home/iulab1/PycharmProjects/nnUNet/utils/config.py) — 🟢 경로 해석 유틸

1. `resolve_project_root()`: 환경변수 → 파일 기반 fallback 체인
2. `resolve_paths(cfg, project_root)`: `${PROJECT_ROOT}` 플레이스홀더 치환

---

## Verification Plan

1. ✅ 실패 프로세스 kill 완료 (PID 991077)
2. wandb offline 모드로 `nnUNetv2_train` 실행
3. **Epoch 0 완료 확인**: `train_loss` ≠ 0.5, `val_loss` ≠ nan, Dice > 0
4. Epoch 1~2 연속 정상 출력 확인 (NaN 미발생)
