---
# Hugging Face Model Card Metadata (YAML front matter)
language:
  - en
license: apache-2.0
tags:
  - image-segmentation
  - semantic-segmentation
  - nnunet
  - medical-imaging
  - eye-segmentation
  - opencv
  - gaze-tracking
  - openeds
datasets:
  - facebook/OpenEDS2019
metrics:
  - dice
pipeline_tag: image-segmentation
library_name: nnunetv2
---

# nnUNet-2D-OpenEDS2019 — Eye Semantic Segmentation

## Model Description

**nnUNet-2D-OpenEDS2019**는 [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) 프레임워크를 기반으로 **OpenEDS 2019** 안구 세그멘테이션 데이터셋에서 파인튜닝된 2D 시맨틱 세그멘테이션 모델입니다. 인코더에 ImageNet 사전학습 ResNet-34 가중치를 Shape-Matching 기법으로 부분 전이한 뒤, Dice + Cross-Entropy 복합 손실 함수를 사용하여 학습하였습니다.

- **Developed by:** hyun02051 (Hongik University)
- **Model type:** 2D PlainConvUNet (7-stage encoder-decoder with deep supervision)
- **Language(s):** N/A (Vision model)
- **License:** Apache 2.0
- **Finetuned from:** ImageNet-pretrained ResNet-34 encoder (partial shape-matching transfer, 11/14 encoder conv layers)
- **Framework:** nnU-Net v2

## Model Sources

- **Repository:** [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- **Dataset:** [OpenEDS 2019 Semantic Segmentation Challenge](https://research.facebook.com/openeds-challenge/)
- **W&B Training Dashboard:** [nnUNet_OpenEDS2019 (wandb.ai)](https://wandb.ai/hyun02051-hongik-university/nnUNet_OpenEDS2019)

---

## Uses

### Direct Use

적외선(IR) 카메라로 촬영한 **안구 이미지**의 4-class 시맨틱 세그멘테이션:

| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | Background | 안구 외 영역 (피부, 속눈썹 등) |
| 1 | Sclera | 공막 (흰자위) |
| 2 | Iris | 홍채 |
| 3 | Pupil | 동공 |

### Downstream Use

- **시선 추적(Gaze Tracking):** Pupil/Iris 세그멘테이션 마스크를 이용하여 동공 중심 좌표 추출 → 시선 방향 추정
- **AR/VR 디바이스:** 헤드마운트 디스플레이 내부 IR 카메라 기반 실시간 눈 추적
- **의료 영상:** 안과 검진 시 홍채/동공 영역 자동 측정

### Out-of-Scope Use

- RGB 카메라로 촬영된 일반 얼굴 이미지 (본 모델은 IR 그레이스케일 이미지 전용)
- 해상도가 400×640px와 크게 다른 이미지
- 안구 질환 진단 (본 모델은 해부학적 구조 세그멘테이션만 수행)

---

## Training Details

### Training Data

**OpenEDS 2019 Semantic Segmentation Dataset**

| Split | Subjects | Segmentation Images |
|-------|----------|---------------------|
| Train | 95 | 8,916 |
| Validation | 28 | 2,403 |
| Test | 29 | 1,440 |

- **총 학습 데이터 (nnUNet imagesTr):** 11,319장 (Train + Validation 통합, nnUNet 내부 5-fold CV 수행)
- **이미지 규격:** 400×640px, 단일 채널 그레이스케일 (IR 카메라)
- **라벨 형식:** 원본 `.npy` → NIfTI `.nii.gz`로 변환 (pseudo-3D, z=1)

#### Class Distribution (학습 세트 기준)

| Class | Pixel Percentage | Imbalance Ratio |
|-------|-----------------|-----------------|
| Background | 83.42% | — |
| Sclera | 7.94% | 10.5× |
| Iris | 7.65% | 10.9× |
| Pupil | 0.99% | 84.3× |

> 극심한 클래스 불균형이 존재하며, 특히 Pupil 클래스가 전체 픽셀의 0.99%에 불과합니다. Dice + CE 복합 손실 함수로 이를 보완하였습니다.

### Training Procedure

#### Architecture

```
PlainConvUNet (2D)
├── Encoder: 7 stages, features [32, 64, 128, 256, 512, 512, 512]
│   ├── Conv2d (3×3) + InstanceNorm2d + LeakyReLU
│   └── 2 conv blocks per stage
├── Decoder: 6 stages with transposed convolutions
│   └── 2 conv blocks per stage
└── Segmentation Head: 1×1 Conv2d → 4 classes
```

#### Preprocessing (nnUNet automated)

| Parameter | Value |
|-----------|-------|
| Patch Size | 448 × 640 |
| Spacing | 1.0 × 1.0 mm |
| Normalization | ZScoreNormalization (μ=86.4, σ=39.9) |
| Resampling | 3rd-order spline (data), 1st-order (seg) |

#### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD (momentum=0.99, nesterov=True) |
| Initial LR | 1e-3 (fine-tuning; default nnUNet uses 1e-2) |
| LR Schedule | PolyLR (power=0.9) |
| Weight Decay | 3e-5 |
| Batch Size | 11 |
| Total Epochs | 1,000 |
| Loss Function | Dice + CrossEntropy (batch dice) |
| Deep Supervision | Enabled |
| Data Augmentation | nnUNet standard (rotation, scaling, elastic, gamma, mirroring) |
| Cross-Validation | 5-fold (fold 0: 9,055 train / 2,264 val) |

#### Pretrained Weight Transfer

ImageNet-pretrained **ResNet-34** 인코더로부터 **Shape-Matching** 기법을 사용하여 Conv2d 가중치를 부분 전이:

| Source (ResNet-34) | Target (nnUNet Encoder) | Weight Shape |
|---|---|---|
| layer1.0.conv1 | encoder.stages.1.0.convs.1.conv | [64, 64, 3, 3] |
| layer2.0.conv1 | encoder.stages.2.0.convs.0.conv | [128, 64, 3, 3] |
| layer2.0.conv2 | encoder.stages.2.0.convs.1.conv | [128, 128, 3, 3] |
| layer3.0.conv1 | encoder.stages.3.0.convs.0.conv | [256, 128, 3, 3] |
| layer3.0.conv2 | encoder.stages.3.0.convs.1.conv | [256, 256, 3, 3] |
| layer4.0.conv1 | encoder.stages.4.0.convs.0.conv | [512, 256, 3, 3] |
| layer4.0.conv2 | encoder.stages.4.0.convs.1.conv | [512, 512, 3, 3] |
| layer4.1.conv1 | encoder.stages.5.0.convs.0.conv | [512, 512, 3, 3] |
| layer4.1.conv2 | encoder.stages.5.0.convs.1.conv | [512, 512, 3, 3] |
| layer4.2.conv1 | encoder.stages.6.0.convs.0.conv | [512, 512, 3, 3] |
| layer4.2.conv2 | encoder.stages.6.0.convs.1.conv | [512, 512, 3, 3] |

- **전이된 레이어:** 11 / 14 (78.6%)
- **전이 불가 레이어:** Stage 0 (채널 불일치: nnUNet 32ch vs ResNet 64ch) 및 첫 번째 conv (입력 채널 불일치: 1ch vs 3ch)

---

## Evaluation

### Metrics

- **Dice Coefficient** (per-class, foreground mean)
- **EMA Pseudo Dice** (nnUNet 내부 지표, exponential moving average)

### Interim Results (학습 진행 중)

> ⚠️ 학습이 아직 완료되지 않았습니다. 아래는 중간 결과입니다.

| Metric | Epoch ~105 (interim) | Final (TBD) |
|--------|---------------------|-------------|
| Train Loss | -0.9476 | |
| Val Loss | -0.9493 | |
| Dice — Sclera | 0.9640 | |
| Dice — Iris | 0.9771 | |
| Dice — Pupil | 0.9728 | |
| EMA FG Dice (mean) | ~0.97 | |

### Final Test Set Results

> 학습 완료 후 `nnUNetv2_predict` 및 `nnUNetv2_evaluate`를 수행하여 Test Set (1,440장) 결과를 여기에 기재할 예정입니다.

| Metric | Sclera | Iris | Pupil | Mean FG |
|--------|--------|------|-------|---------|
| Dice |  |  |  |  |
| IoU |  |  |  |  |
| HD95 |  |  |  |  |

---

## Technical Specifications

### Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 5090 (32 GB VRAM) |
| VRAM Usage | ~7.5 GB |
| GPU Utilization | ~97% |
| Epoch Time | ~15 seconds |
| Estimated Total Training Time | ~4.2 hours (1,000 epochs) |

### Software

| Component | Version |
|-----------|---------|
| nnU-Net | v2 (dev branch, MIC-DKFZ/nnUNet) |
| PyTorch | 2.x (with torch.compile enabled) |
| CUDA | 13.3 (Driver 610.43) |
| Python | 3.10 |
| torchvision | (for ResNet-34 pretrained weights) |
| wandb | 0.28.1 |

---

## How to Use

### Inference with nnUNetv2

```bash
# Set environment variables
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# Run prediction
nnUNetv2_predict \
    -i /path/to/input_images \
    -o /path/to/output_predictions \
    -d 600 \
    -c 2d \
    -f 0 \
    -tr nnUNetTrainer_ImageNetPretrained
```

### Resume Training

```bash
nnUNetv2_train 600 2d 0 \
    -tr nnUNetTrainer_ImageNetPretrained \
    --c
```

---

## Citation

### nnU-Net

```bibtex
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

### OpenEDS 2019

```bibtex
@inproceedings{garbin2019openeds,
  title={OpenEDS: Open Eye Dataset},
  author={Garbin, Stephan J and Shen, Yiru and Schuber, Immo and Roth, Robert and Kang, Chin-Ling and Bejnordi, Babak Ehteshami and Ma, Junfeng and Lohit, Suhas and Marber, Sarah and Jovanovic, Nenad and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2019}
}
```

### ResNet (Pretrained Encoder)

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

---

## Model Card Contact

- **Author:** hyun02051
- **Affiliation:** Hongik University
- **W&B:** [hyun02051-hongik-university](https://wandb.ai/hyun02051-hongik-university)
