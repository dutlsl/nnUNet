# Epoch 201 NaN 발생 근본 원인 분석 및 원인 규명 보고서

> **문서 목적**: 싱글 GPU 학습 세션 중 Epoch 201 시점부터 `val_loss: nan` 및 `train_loss: 0.5000` 고정 현상이 발생한 원인을 소스 코드 차원에서 정밀 분석하고 원인을 명확하게 보고합니다.

---

## 📌 Executive Summary

1. **원인 1: AMP FP16 연산 중 `KL Divergence` 및 `InfoNCE` 오버플로우/언더플로우**:
   - `DomainConsistencyLoss`에서 `torch.log(p_soft + 1e-8)` 계산 시, PyTorch AMP FP16 정밀도 한계($6 \times 10^{-5}$)로 인해 `1e-8`이 언더플로우되어 `torch.log(0.0) = -inf`가 발생했습니다.
   - `StructuralContrastiveLoss`에서 `sim_matrix / self.temperature` (temperature = 0.07) 나누기 연산 시, FP16 지수 한계(65504)를 넘어 오버플로우 `+inf`가 발생했습니다.

2. **원인 2: `torch.nan_to_num(l, nan=0.5)` 함정 (오류 은폐 및 파라미터 파괴)**:
   - `nnUNetTrainer_Vivim_SADG.py`의 `train_step`에서 `NaN` 발생 시 `torch.nan_to_num(l, nan=0.5)`로 가짜 loss `0.5`를 부여했습니다.
   - 이로 인해 역전파 시 `inf/nan` 경사(Gradient)가 가중치에 그대로 반영되어 Epoch 201 시점에 신경망의 모든 가중치가 붕괴(Corrupted)되고 `val_loss`가 `NaN`으로 박힌 채 `train_loss`가 `0.5`로 굳어버렸습니다.

---

## 1. 코드 기반 근본 원인 1: AMP FP16 언더플로우 / 오버플로우

### 가. `DomainConsistencyLoss` (KL Divergence Log 언더플로우)

[losses/sadg_loss.py](file:///home/iulab1/PycharmProjects/nnUNet/losses/sadg_loss.py#L128-L132)

```python
# losses/sadg_loss.py (수정 전 코드)
kl_pa = F.kl_div(
    torch.log(p_soft + 1e-8), a_soft, reduction='none'
).sum(dim=1).mean()
```

- PyTorch Automatic Mixed Precision (AMP FP16) 상태에서는 최소 표현 가능 수치가 약 $6 \times 10^{-5}$입니다.
- `p_soft`가 0에 가깝고 `1e-8`을 더하면 FP16 정밀도 표현 범위를 벗어나 `0.0`으로 잘려나갑니다 (Underflow).
- 그 결과 `torch.log(0.0) = -inf`가 계산되고, `F.kl_div` 손실 결과가 **`NaN` / `-inf`**로 붕괴됩니다.

### 나. `StructuralContrastiveLoss` (InfoNCE Division 오버플로우)

[losses/sadg_loss.py](file:///home/iulab1/PycharmProjects/nnUNet/losses/sadg_loss.py#L185-L187)

```python
# losses/sadg_loss.py (수정 전 코드)
sim_matrix = torch.bmm(anchors, a_norm.transpose(1, 2)) / self.temperature  # self.temperature = 0.07
sim_matrix = torch.clamp(sim_matrix, min=-50.0, max=50.0)
```

- `anchors`와 `a_norm`의 내적값을 `0.07`이라는 작은 temperature로 나눌 때 FP16 상태에서 지수 표현 범위를 넘어서는 오버플로우(`+inf`)가 발생합니다.
- `clamp` 이전에 이미 `+inf`가 생성되어 이후 `nan`으로 전환됩니다.

---

## 2. 코드 기반 근본 원인 2: `nan_to_num(0.5)` 손실 은폐 함정

[nnUNetTrainer_Vivim_SADG.py](file:///home/iulab1/PycharmProjects/nnUNet/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainer_Vivim_SADG.py#L508-L510)

```python
# nnUNetTrainer_Vivim_SADG.py (수정 전 코드)
l = loss_dict['total']
if torch.isnan(l) or torch.isinf(l):
    l = torch.nan_to_num(l, nan=0.5, posinf=1.0, neginf=0.0) # ❌ 치명적 오류 은폐!
```

- `NaN`이나 `Inf`가 발생했을 때 학습 스텝을 즉시 건너뛰고(Skip) 가중치를 보호해야 하지만, `nan_to_num`으로 가짜 손실값 `0.5`를 덮어씌워 역전파를 강제 진행했습니다.
- 오염된 경사(Gradient)가 가중치에 지속적으로 누적되면서 백본 및 SADG 모듈 파라미터 전체가 붕괴되어:
  - `val_loss`: `NaN`으로 굳어짐
  - `train_loss`: 가짜 값 `0.5000`으로 굳어짐
  - `Dice Score`: `0.0000`으로 폭망함

---

## 3. 해결 조치 사항 요약

1. **FP32 안전 연산 보장**:
   - `DomainConsistencyLoss` 및 `StructuralContrastiveLoss` 연산 시 `.float()`로 FP32 변환 후 `clamp(min=1e-6)` 적용하여 FP16 언더플로우/오버플로우 방지.
2. **안전한 Step-Skipping**:
   - `train_step`에서 `NaN/Inf` 발생 시 가짜 `0.5` 값을 부여하지 않고 `self.optimizer.zero_grad()`로 해당 스텝을 안전하게 넘기도록 조치.

원인 분석 보고서를 확인해 주시기 바랍니다.
