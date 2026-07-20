"""
Run Full Test Set Inference for Dataset600_OpenEDS2019 using checkpoint_best.pth
"""
import os
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main():
    os.environ['nnUNet_raw'] = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_raw'
    os.environ['nnUNet_preprocessed'] = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_preprocessed'
    os.environ['nnUNet_results'] = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results'

    model_dir = '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_ImageNetPretrained__nnUNetPlans__2d'
    output_dir = os.path.join(model_dir, 'test_predictions')
    os.makedirs(output_dir, exist_ok=True)

    print(f"==================================================")
    print(f"Starting Full Test Set Prediction (1,440 images)")
    print(f"Model Folder : {model_dir}")
    print(f"Checkpoint   : checkpoint_best.pth")
    print(f"Output Dir   : {output_dir}")
    print(f"==================================================")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda'),
        verbose=False
    )

    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth'
    )

    predictor.predict_from_files(
        '/home/iulab0/PycharmProjects/nnUNet/nnUNet_raw/Dataset600_OpenEDS2019/imagesTs',
        output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=4,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

    print(f"\n==================================================")
    print(f"Full Test Set Prediction Successfully Completed!")
    print(f"==================================================")

if __name__ == '__main__':
    main()
