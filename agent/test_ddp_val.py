import torch.multiprocessing as mp
import torch, sys, json, os, numpy as np
sys.path.insert(0, '/home/iulab1/PycharmProjects/nnUNet')
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_Vivim_SADG import nnUNetTrainer_Vivim_SADG

def run_test(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29519'
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)

    with open('/home/iulab1/PycharmProjects/nnUNet/nnUNet_preprocessed/Dataset600_OpenEDS2019/nnUNetPlans.json') as f:
        plans = json.load(f)
    plans['continue_training'] = False
    dataset_json = {'name': 'OpenEDS', 'labels': {'background': 0, 'pupil': 1, 'iris': 2, 'sclera': 3}, 'file_ending': '.png'}

    trainer = nnUNetTrainer_Vivim_SADG(plans, '2d', 0, dataset_json)
    trainer.is_ddp = True
    trainer.local_rank = rank
    trainer.device = torch.device('cuda', rank)
    trainer.initial_lr = 3e-4
    trainer.initialize()
    trainer.on_train_start()

    val_outputs = []
    for i, batch in enumerate(trainer.dataloader_val):
        if i >= 5:
            break
        with torch.no_grad():
            res = trainer.validation_step(batch)
            val_outputs.append(res)

    print(f'Rank {rank} val_outputs len:', len(val_outputs), 'batch 0 loss:', val_outputs[0]['loss'].shape, 'tp:', val_outputs[0]['tp_hard'], flush=True)
    trainer.on_validation_epoch_end(val_outputs)
    print(f'Rank {rank} logged dice:', trainer.logger.get_value('dice_per_class_or_region', step=None), flush=True)
    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run_test, args=(2,), nprocs=2)
