#!/bin/bash
export PYTHONPATH=/home/iulab0/PycharmProjects/nnUNet
export nnUNet_raw=/home/iulab0/PycharmProjects/nnUNet/nnUNet_raw
export nnUNet_preprocessed=/home/iulab0/PycharmProjects/nnUNet/nnUNet_preprocessed
export nnUNet_results=/home/iulab0/PycharmProjects/nnUNet/nnUNet_results
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0
export OPENCV_LOG_LEVEL=OFF
export OPENCV_VIDEOIO_DEBUG=0

/home/iulab0/PycharmProjects/nnUNet/nnunet_env/bin/python -u /home/iulab0/PycharmProjects/nnUNet/nnunetv2/run/run_training.py Dataset600_OpenEDS2019 2d 0 -tr nnUNetTrainer_Vivim_SADG
