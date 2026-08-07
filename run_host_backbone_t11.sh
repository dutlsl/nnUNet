#!/bin/bash
export PYTHONPATH=/home/iulab0/PycharmProjects/nnUNet
export nnUNet_raw=/home/iulab0/PycharmProjects/nnUNet/nnUNet_raw
export nnUNet_preprocessed=/home/iulab0/PycharmProjects/nnUNet/nnUNet_preprocessed
export nnUNet_results=/home/iulab0/PycharmProjects/nnUNet/nnUNet_results
export TEMPORAL_WINDOW=11
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0
export nnUNet_compile=false
export OPENCV_LOG_LEVEL=OFF
export OPENCV_VIDEOIO_DEBUG=0

# Wait for T=9 training process (PID 822355) to finish
echo "Waiting for T=9 training process (PID 822355) to finish..."
while kill -0 822355 2>/dev/null; do
    sleep 10
done

echo "T=9 training process finished! Starting T=11 training..."
/home/iulab0/PycharmProjects/nnUNet/nnunet_env/bin/python -u /home/iulab0/PycharmProjects/nnUNet/nnunetv2/run/run_training.py Dataset600_OpenEDS2019 2d 1 -tr nnUNetTrainer_Vivim > logs/host_backbone_t11_train.log 2>&1
