# OpenEDS 2019 안구 데이터셋 종합 분석 보고서

본 문서는 **OpenEDS 2019 (Open Eye Data Set)** 전체 데이터셋에 대한 정밀 분석 결과입니다.
에이전트가 데이터 관련 의사결정 시 참고할 수 있도록 작성되었습니다.

> [!IMPORTANT]
> **데이터셋 루트 경로**: `/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/`

---

## 0. OpenEDS 2019 전체 개요

**OpenEDS**는 VR 헤드셋(HMD) 환경에서 동기화된 2개의 적외선(IR) 안구 카메라로 수집된 공개 안구 이미지 데이터셋입니다.

- **참여자**: 총 152명
- **이미지 해상도**: 400 × 640 픽셀, 그레이스케일
- **조명**: 통제된 IR 조명 환경
- **라이선스**: CC BY-NC 4.0 (Facebook Technologies, LLC)
- **논문**: `Openedsdata2019/openeds_paper.pdf`

### 데이터셋 구성 (3개 서브셋)

| 서브셋 | 총 이미지 수 | 라벨 유무 | 파일 경로 | 주요 용도 |
| :--- | :---: | :---: | :--- | :--- |
| **Semantic Segmentation** | 12,759장 | ✅ 4-class 마스크 | `Semantic_Segmentation_Dataset/` | 지도 학습 (세그멘테이션) |
| **Generative** | 252,690장 | ❌ | `Generative_Dataset.zip` (미해제) | GAN / 비지도 학습 |
| **Sequence** | 91,200장 | ❌ | `Sequence_Dataset/Sequence_Dataset/` | 시간적 동역학 / Self-supervised |

### 메타데이터 JSON 파일

| 파일명 | 경로 |
| :--- | :--- |
| Train mapping | `OpenEDS_train_userID_mapping_to_images.json` |
| Validation mapping | `OpenEDS_validation_userID_mapping_to_images.json` |
| Test mapping | `OpenEDS_test_userID_mapping_to_images.json` |

각 JSON은 `{ "subject_id": { "segmentation": [...], "generative": [...], "sequence": [...] } }` 형태로 피험자별 이미지 파일명 매핑을 제공합니다.

---

## 1. 피험자 분포 및 데이터 분할 (Subject Statistics)

OpenEDS 데이터셋은 학습/검증/테스트 그룹 간 **피험자가 중복되지 않도록 Subject-level Split**되어 있습니다.

### A. 분할별 요약

| Split | Subjects | Seg Images | Mean Seg/Subj | Min/Max Seg | Gen Images | Seq Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 95 | 8,916 | 93.85 | 1 / 222 | 193,882 | 57,000 |
| **Validation** | 28 | 2,403 | 85.82 | 37 / 138 | 57,337 | 16,800 |
| **Test** | 29 | 1,440 | 49.66 | 36 / 80 | 1,471 | 17,400 |
| **합계** | **152** | **12,759** | — | — | **252,690** | **91,200** |

### B. 피험자별 통계 (Pandas Aggregation)

| Split | Seg mean | Seg min | Seg max | Seg std | Gen sum | Seq sum | Seq mean/subj |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Train | 93.85 | 1 | 222 | 27.97 | 193,882 | 57,000 | 600 |
| Validation | 85.82 | 37 | 138 | 22.71 | 57,337 | 16,800 | 600 |
| Test | 49.66 | 36 | 80 | 12.26 | 1,471 | 17,400 | 600 |

---

## 2. Semantic Segmentation 데이터셋 상세 분석

### 경로 구조

```
Openedsdata2019/Semantic_Segmentation_Dataset/
├── train/        # 8,916장 (images + labels)
├── validation/   # 2,403장
└── test/         # 1,440장
```

nnUNet 변환 후 경로:
```
nnUNet_raw/Dataset600_OpenEDS2019/
├── imagesTr/     # 학습 이미지 (.png → .nii.gz 변환)
├── labelsTr/     # 학습 레이블
├── imagesTs/     # 테스트 이미지
├── labelsTs/     # 테스트 레이블
└── dataset.json  # nnUNet 데이터셋 정의
```

### 클래스별 픽셀 분포 (500개 샘플 기반)

| Class ID | Class Name | Pixel Count | Percentage |
| :---: | :--- | :---: | :---: |
| **0** | Background (배경) | 106,780,512 | **83.42%** |
| **1** | Sclera (공막/흰자위) | 10,160,845 | **7.94%** |
| **2** | Iris (홍채) | 9,787,568 | **7.65%** |
| **3** | Pupil (동공) | 1,271,075 | **0.99%** |

> [!WARNING]
> **극심한 클래스 불균형**: Background가 83.42%를 차지. Pupil은 0.99%로 매우 미미.
> nnUNetv2의 Dice + CE 복합 손실과 Deep Supervision이 이 불균형 극복의 핵심.

### 이미지 밝기(Intensity) 통계

| 항목 | 값 |
| :--- | :--- |
| 해상도 | 400 × 640 픽셀 (그레이스케일) |
| 평균 최소 밝기 (Min) | 1.11 |
| 평균 최대 밝기 (Max) | 255.00 |
| 전체 평균 밝기 (Mean) | 117.46 |
| 전체 표준편차 (Std) | 62.78 |

- IR 조명 특성으로 안구 영역만 밝고, 주변 피부는 거의 검정에 가까움.
- nnUNetv2는 이 통계를 자동 감지하여 **Z-score 정규화** 적용.

### 현재 nnUNet 학습 설정

| 항목 | 값 |
| :--- | :--- |
| Dataset ID | 600 (Dataset600_OpenEDS2019) |
| Trainer | `nnUNetTrainer_ImageNetPretrained` |
| Configuration | 2d |
| Fold | 0 |
| Architecture | PlainConvUNet (7 stages, features [32,64,128,256,512,512,512]) |
| Patch Size | [448, 640] |
| Batch Size | 11 |
| Initial LR | 1e-3 (pretrained fine-tuning) |
| Normalization | ZScoreNormalization |
| Results Path | `nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d/fold_0/` |

---

## 3. Sequence Dataset 상세 분석

### 개요

Sequence Dataset은 VR HMD에서 **200 Hz (초당 200프레임)** 으로 촬영된 고속 연속 안구 이미지입니다.
피험자 1인당 600프레임(= 3.0초 분량)이 할당되어 있으며, **레이블(마스크) 없이** 이미지만 제공됩니다.

### 경로 구조

```
Openedsdata2019/Sequence_Dataset/Sequence_Dataset/
├── train/
│   ├── Train_Part_1.tar.gz   (약 1.25 GB)
│   ├── Train_Part_2.tar.gz
│   ├── Train_Part_3.tar.gz
│   ├── Train_Part_4.tar.gz
│   ├── Train_Part_5.tar.gz
│   └── Train_Part_6.tar.gz
├── validation/
│   ├── Validation_Part_1.tar.gz   (약 1.1 GB)
│   └── Validation_Part_2.tar.gz
└── test/
    ├── Test_Part_1.tar.gz   (약 1.16 GB)
    └── Test_Part_2.tar.gz
```

> [!NOTE]
> 현재 zip 압축은 해제 완료되었으나, 각 split 내부의 `.tar.gz` 파일은 **아직 추가 해제가 필요**합니다.
> 내부 파일은 `{12자리 숫자}.png` 형태의 400×640 그레이스케일 이미지입니다.

### 분할별 프레임 수

| Split | Subjects | 총 프레임 수 | 피험자당 프레임 | tar.gz 파트 수 | 총 용량 (압축) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 95 | 57,000 | 600 | 6 | 약 7.5 GB |
| **Validation** | 28 | 16,800 | 600 | 2 | 약 2.2 GB |
| **Test** | 29 | 17,400 | 600 | 2 | 약 2.3 GB |
| **합계** | **152** | **91,200** | **600** | **10** | **약 12.0 GB** |

### 이미지 사양

| 항목 | 값 |
| :--- | :--- |
| 해상도 | 400 × 640 픽셀 |
| 색상 | 그레이스케일 (단일 채널) |
| 파일 형식 | PNG |
| 파일 크기 (개당) | 약 110 ~ 145 KB |
| 프레임 레이트 | 200 Hz |
| 촬영 시간/피험자 | 약 3.0초 (600 frames ÷ 200 Hz) |

### 활용 분야

1. **자기지도 학습 (Self-Supervised Pre-training)**
   - Video MAE, DINO, SimCLR, MoCo 등을 활용한 시각적 표현 사전 학습
   - 세그멘테이션 인코더의 초기화 가중치로 전이 학습 가능

2. **안구 운동 시간적 동역학 분석 (Temporal Dynamics)**
   - 200 Hz 고해상도로 사케드(Saccade), 미세 사케드(Micro-saccade), 깜빡임(Blink) 모델링
   - 안구 운동 분류기 학습 데이터

3. **시선 추적 (Gaze Tracking) / Optical Flow**
   - 연속 프레임 간 모션 벡터 추정
   - 시선 이동 궤적 예측 모델 개발

4. **준지도 학습 (Semi-supervised Learning)**
   - 소량 라벨 데이터(Seg) + 대량 미라벨 데이터(Seq/Gen) 결합
   - Mean Teacher, FixMatch 등 적용 가능

---

## 4. 파일 경로 퀵 레퍼런스

| 리소스 | 절대 경로 |
| :--- | :--- |
| 데이터셋 루트 | `/home/iulab0/PycharmProjects/nnUNet/Openedsdata2019/` |
| Seg Dataset (원본) | `Openedsdata2019/Semantic_Segmentation_Dataset/` |
| Seg Test Dataset (원본) | `Openedsdata2019/Semantic_Segmentation_Test_Dataset/` |
| Seq Dataset (압축 해제) | `Openedsdata2019/Sequence_Dataset/Sequence_Dataset/` |
| Gen Dataset (미해제) | `Openedsdata2019/Generative_Dataset.zip` |
| nnUNet raw | `/home/iulab0/PycharmProjects/nnUNet/nnUNet_raw/Dataset600_OpenEDS2019/` |
| nnUNet preprocessed | `/home/iulab0/PycharmProjects/nnUNet/nnUNet_preprocessed/Dataset600_OpenEDS2019/` |
| nnUNet results | `/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/Dataset600_OpenEDS2019/` |
| 변환 스크립트 | `/home/iulab0/PycharmProjects/nnUNet/agent/convert_openeds_to_nnunet.py` |
| README | `Openedsdata2019/README.pdf` |
| 논문 | `Openedsdata2019/openeds_paper.pdf` |
