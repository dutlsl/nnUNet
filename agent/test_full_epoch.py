import sys, json, torch, os, numpy as np
os.environ['WANDB_MODE'] = 'disabled'
sys.path.insert(0, '/home/iulab1/PycharmProjects/nnUNet')
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_Vivim_SADG import nnUNetTrainer_Vivim_SADG

with open('/home/iulab1/PycharmProjects/nnUNet/nnUNet_preprocessed/Dataset600_OpenEDS2019/nnUNetPlans.json') as f:
    plans = json.load(f)
plans['continue_training'] = False
dataset_json = {'name': 'OpenEDS', 'labels': {'background': 0, 'pupil': 1, 'iris': 2, 'sclera': 3}, 'file_ending': '.png'}

trainer = nnUNetTrainer_Vivim_SADG(plans, '2d', 0, dataset_json)
trainer.initial_lr = 3e-4
trainer.initialize()
trainer.on_train_start()

print("Running FULL 250 training steps of Epoch 0 with dynamic SAS + SoftDice...", flush=True)
for i in range(250):
    res = trainer.train_step(next(iter(trainer.dataloader_train)))
    if (i + 1) % 50 == 0 or i == 0:
        print(f"Step {i+1}/250 loss: {res['loss']:.4f}", flush=True)

print("Running validation step...", flush=True)
val_outputs = []
for i, batch in enumerate(trainer.dataloader_val):
    with torch.no_grad():
        val_outputs.append(trainer.validation_step(batch))

trainer.on_validation_epoch_end(val_outputs)
dice_per_class = trainer.logger.get_value('dice_per_class_or_region', step=None)
print(f"FULL EPOCH 0 VALIDATION DICE: {dice_per_class}", flush=True)
