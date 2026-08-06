"""
nnUNetTrainer_Vivim_SADG: nnUNet v2 Trainer for Vivim + SAGD.
Integrates Structure-Aware Serialization (SAS), Hierarchical Domain Modeling (HDM),
Spectral Graph Alignment (SGA), and multi-domain training (OpenEDS + Swirski + LPW).

All configurations come from configs/sadg_vivim.yaml.
Dual GPU training via DistributedDataParallel.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime

# Ensure project root is in sys.path
# File is at: nnunetv2/training/nnUNetTrainer/variants/network_architecture/
# 6 levels up -> nnUNet project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from torch.nn.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from models.vivim_backbone import VivimBackbone
from models.sadg_serialization import StructureAwareSerializer
from models.sadg_hdm import HDM2D
from models.sadg_sga import SpectralGraphAlignment
from losses.sadg_loss import SAGDLoss, DiceCELoss
from datasets.multi_domain_dataset import (
    get_multi_domain_dataloaders,
    collate_multi_domain,
    RITnetPreprocessor,
    OpenEDSDomainDataset,
)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from utils.config import ConfigDict, load_config
from utils.metrics import compute_dice_score


class VivimSAGDWrapper(nn.Module):
    """
    nnUNet-compatible wrapper for Vivim + SAGD modules.
    Manages SAS, HDM, SGA as sub-modules alongside the Vivim backbone.
    """

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg.model

        self.backbone = VivimBackbone(
            in_channels=1,
            num_classes=model_cfg.num_classes,
            base_channels=model_cfg.base_channels,
            d_state=model_cfg.mamba.d_state,
            d_conv=model_cfg.mamba.d_conv,
            expand=model_cfg.mamba.expand,
            use_mamba=cfg.ablation.use_mamba,
            use_sadg=cfg.ablation.use_sadg,
        )

        # SAS module
        if cfg.ablation.use_sas:
            self.sas = StructureAwareSerializer(cfg)
            self.backbone.sas = self.sas
        else:
            self.sas = None

        # HDM module
        if cfg.ablation.use_hdm:
            self.hdm = HDM2D(cfg)
            self.backbone.hdm = self.hdm
        else:
            self.hdm = None

        # SGA module
        if cfg.ablation.use_sga:
            self.sga = SpectralGraphAlignment(
                num_classes=model_cfg.num_classes,
                feature_dim=model_cfg.base_channels * 8,
                num_prototypes_per_class=model_cfg.sga.num_prototypes_per_class,
                temperature=model_cfg.sga.alignment_temperature,
                spectral_k=model_cfg.sga.spectral_k,
                momentum=model_cfg.sga.prototype_momentum,
            )
            self.backbone.sga = self.sga
        else:
            self.sga = None

        self.cfg = cfg

    def forward(self, x, labels=None):
        """Standard forward (or multi-domain training if x is dict)."""
        if isinstance(x, dict):
            return self.forward_multi_domain(x)
        if x.ndim == 4:
            x = x.unsqueeze(1)  # [B, C, H, W] -> [B, 1, C, H, W]
        return self.backbone(x, labels=labels)

    def forward_multi_domain(self, domain_batches):
        """
        Multi-domain forward for training with HDM.

        Args:
            domain_batches: dict[domain_id] -> {'images': [B, T, C, H, W], 'label': [B, H, W]}

        Returns:
            dict with primary domain 'seg_logits', per-domain 'serialized_tokens', etc.
        """
        domain_outputs = {}
        serialized_list = []
        domain_ids = []

        for d_id in sorted(domain_batches.keys()):
            batch = domain_batches[d_id]
            images = batch['images']  # [B, T, C, H, W]

            out = self.backbone.forward_domain(images, domain_id=d_id)
            domain_outputs[d_id] = out

            if out['serialized_tokens'] is not None:
                serialized_list.append(out['serialized_tokens'])
                domain_ids.append(d_id)

        # HDM: hierarchical domain modeling (always on primary GPU)
        if self.hdm is not None and len(serialized_list) > 1 and self.training:
            hdm_output = self.hdm(serialized_list, domain_ids)

            # SGA: update prototypes with HDM-fused tokens
            primary_out = domain_outputs[0]
            if self.sga is not None and 0 in domain_batches:
                primary_labels = domain_batches[0]['label']
                hdm_output = self.sga(
                    hdm_output,
                    primary_labels,
                    inv_cds_order=primary_out.get('inv_cds_order', None)
                )

            # Decode primary domain from HDM-fused tokens
            primary_seg = self.backbone.decode_from_tokens(
                hdm_output,
                primary_out['skips'],
                primary_out['spatial_shape'],
                inv_cds_order=primary_out.get('inv_cds_order', None),
            )
        elif 0 in domain_outputs:
            # Single domain or no HDM
            primary_out = domain_outputs[0]
            tokens = primary_out.get('serialized_tokens', None)

            if tokens is not None:
                if self.sga is not None:
                    tokens = self.sga(
                        tokens,
                        domain_batches[0].get('label', None),
                        inv_cds_order=primary_out.get('inv_cds_order', None)
                    )
                primary_seg = self.backbone.decode_from_tokens(
                    tokens,
                    primary_out['skips'],
                    primary_out['spatial_shape'],
                    inv_cds_order=primary_out.get('inv_cds_order', None),
                )
            else:
                primary_seg = self.backbone.decode_from_tokens(
                    primary_out['bottleneck'].reshape(
                        primary_out['bottleneck'].shape[0],
                        primary_out['bottleneck'].shape[1],
                        -1
                    ).permute(0, 2, 1),
                    primary_out['skips'],
                    primary_out['spatial_shape'],
                )
        else:
            raise ValueError("Primary domain (0) not in domain_batches")

        # Auxiliary domains: decode auxiliary ISM outputs into seg logits
        # for DomainConsistencyLoss (KL divergence between primary and auxiliary predictions)
        aux_seg_list = []
        for d_id in sorted(domain_batches.keys()):
            if d_id == 0:
                continue
            if d_id in domain_outputs:
                aux_out = domain_outputs[d_id]
                aux_tokens = aux_out.get('serialized_tokens', None)
                if aux_tokens is not None:
                    aux_seg = self.backbone.decode_from_tokens(
                        aux_tokens,
                        aux_out['skips'],
                        aux_out['spatial_shape'],
                        inv_cds_order=aux_out.get('inv_cds_order', None),
                    )
                    aux_seg_list.append(aux_seg)

        # Gather all serialized tokens for contrastive loss
        primary_tokens = None
        aux_tokens_list = []
        if serialized_list:
            primary_tokens = serialized_list[0] if domain_ids[0] == 0 else None
            for i, d_id in enumerate(domain_ids):
                if d_id != 0:
                    aux_tokens_list.append(serialized_list[i])

        return {
            'seg_logits': primary_seg,
            'primary_tokens': primary_tokens,
            'aux_tokens_list': aux_tokens_list,
            'aux_seg_list': aux_seg_list,
        }


class MultiDomainDataLoaderWrapper:
    """
    Wraps multi-domain DataLoader to yield domain-grouped batches.
    Compatible with nnUNet's train loop expecting __iter__ / __next__.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)

        # batch is dict[domain_id] -> {'images': Tensor, 'label': Tensor, 'domain_id': int}
        # Move to device
        for d_id in batch:
            batch[d_id]['images'] = batch[d_id]['images'].to(self.device, non_blocking=True)
            batch[d_id]['label'] = batch[d_id]['label'].to(self.device, non_blocking=True)

        return batch


class SingleDomainDataLoaderWrapper:
    """Wraps a single-domain DataLoader for validation."""

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)

        return {
            'data': batch['images'].to(self.device, non_blocking=True),
            'target': batch['label'].unsqueeze(1).to(self.device, non_blocking=True),
        }


class nnUNetTrainer_Vivim_SADG(nnUNetTrainer):
    """
    nnUNet v2 Trainer for Vivim + SAGD.
    1000 epochs, PolyLR, AdamW, multi-domain training with HDM.
    Dual GPU training via DataParallel (simpler than DDP for 2-GPU setup).
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda'),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50

        # Fast validation mode: 1 iteration per epoch for rapid 1000-epoch stress test
        if os.environ.get('SAGD_FAST_VALIDATE', '0') == '1':
            self.num_iterations_per_epoch = 1
            self.num_val_iterations_per_epoch = 1
            print("[SAGD] ⚡ FAST VALIDATE MODE: 1 iter/epoch for rapid 1000-epoch NaN stress test", flush=True)
        # Load SAGD config - resolve project root dynamically
        _project_root = os.environ.get(
            'NNUNET_PROJECT_ROOT',
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
        )
        config_path = os.path.join(_project_root, 'configs', 'sadg_vivim.yaml')
        self.sadg_cfg = load_config(config_path)
        self._project_root = _project_root

        # --- Native Dynamic VRAM Planning for Vivim + SADG ---
        plan_patch_voxels = float(np.prod(self.configuration_manager.patch_size))
        model_patch_size = list(self.sadg_cfg.data.input_resolution)
        model_patch_voxels = float(np.prod(model_patch_size))

        vram_scaling_factor = plan_patch_voxels / max(model_patch_voxels, 1.0)
        raw_dynamic_bs = int(round(self.configuration_manager.batch_size * vram_scaling_factor))

        num_domains = getattr(self.sadg_cfg.model.hdm, 'num_domains', 3)
        cfg_bs = getattr(self.sadg_cfg.training, 'batch_size', None)
        if cfg_bs is not None and cfg_bs > 0:
            target_bs = cfg_bs
        else:
            target_bs = raw_dynamic_bs

        dynamic_batch_size = max((target_bs // num_domains) * num_domains, num_domains)

        # Update configuration_manager dynamically
        self.configuration_manager.configuration['patch_size'] = model_patch_size
        self.configuration_manager.configuration['batch_size'] = dynamic_batch_size

        # Register network architecture parameters into nnUNet ConfigurationManager natively
        self.configuration_manager.configuration['architecture'] = {
            'network_class_name': 'VivimSAGDWrapper',
            'arch_kwargs': {
                'num_input_channels': self.num_input_channels,
                'num_output_channels': self.label_manager.num_segmentation_heads,
                'base_channels': self.sadg_cfg.model.base_channels,
                'd_state': self.sadg_cfg.model.mamba.d_state,
                'd_conv': self.sadg_cfg.model.mamba.d_conv,
                'expand': self.sadg_cfg.model.mamba.expand,
                'patch_size': list(self.configuration_manager.patch_size),
                'batch_size': self.configuration_manager.batch_size,
            }
        }

        # Loss
        self.sadg_loss = None  # Initialized in _build_loss

    def _do_i_compile(self):
        return False

    def initialize(self):
        if not self.was_initialized:
            self.initialize_network()
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank], find_unused_parameters=True)
            self.loss = self._build_loss()
            self.dataset_class = None
            self.was_initialized = True

            logger_config_hparas = {
                "initial_lr": self.initial_lr,
                "weight_decay": self.weight_decay,
                "oversample_foreground_percent": self.oversample_foreground_percent,
                "probabilistic_oversampling": self.probabilistic_oversampling,
                "num_iterations_per_epoch": self.num_iterations_per_epoch,
                "num_val_iterations_per_epoch": self.num_val_iterations_per_epoch,
                "num_epochs": self.num_epochs,
                "enable_deep_supervision": self.enable_deep_supervision,
                "batch_size": self.configuration_manager.batch_size
            }
            self.logger.update_config({"hparas": logger_config_hparas})

    def configure_optimizers(self):
        """AdamW with PolyLR scheduler tuned for pretrained Mamba backbone (3e-4)."""
        self.initial_lr = 3e-4
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def initialize_network(self):
        """Build Vivim + SAGD network."""
        self.network = self.build_network_architecture(
            self.plans_manager,
            self.configuration_manager,
            self.num_input_channels,
            self.label_manager.num_segmentation_heads,
            self.enable_deep_supervision
        ).to(self.device)

        # Try to load pretrained weights from baseline Vivim (config-based path resolution)
        ckpt_paths = [
            os.path.join(self._project_root, 'nnUNet_results', 'Dataset600_OpenEDS2019',
                         'nnUNetTrainer_Vivim__nnUNetPlans__2d', 'fold_0', 'checkpoint_final.pth'),
        ]
        # Add config-specified pretrained checkpoint if available
        pretrained_from_cfg = getattr(getattr(self.sadg_cfg, 'training', None), 'pretrained_checkpoint', None)
        if pretrained_from_cfg:
            resolved = pretrained_from_cfg.replace('${PROJECT_ROOT}', self._project_root)
            ckpt_paths.insert(0, resolved)

        for ckpt_path in ckpt_paths:
            if os.path.exists(ckpt_path):
                print(f"[SAGD] Loading pretrained weights from {ckpt_path}...", flush=True)
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                state_dict = ckpt.get('model_state_dict', ckpt.get('network_weights', ckpt))

                if not isinstance(state_dict, dict):
                    continue

                model_keys = set(self.network.state_dict().keys())
                new_state_dict = {}
                for k, v in state_dict.items():
                    if any(skip in k for skip in ['sphere_head', 'weakmed', 'eyeball_head']):
                        continue
                    
                    # Match exact key or strip/add prefixes for DDP and backbone
                    candidates = [
                        k,
                        f"backbone.{k}",
                        f"module.{k}",
                        f"module.backbone.{k}",
                    ]
                    if k.startswith("backbone."):
                        candidates.append(k[9:])
                        candidates.append(f"module.{k[9:]}")
                        candidates.append(f"module.{k}")
                    if k.startswith("module."):
                        candidates.append(k[7:])

                    for cand in candidates:
                        if cand in model_keys:
                            new_state_dict[cand] = v
                            break

                if new_state_dict:
                    missing, unexpected = self.network.load_state_dict(
                        new_state_dict, strict=False
                    )
                    print(f"[SAGD] Successfully loaded {len(new_state_dict)} pretrained weights into network! Missing: {len(missing)}", flush=True)
                    break

    def _build_loss(self):
        """Build SAGD combined loss and validation loss."""
        self.sadg_loss = SAGDLoss(self.sadg_cfg)
        # Cache validation loss function (was being re-instantiated every val step)
        self.val_loss_fn = DiceCELoss(
            num_classes=self.sadg_cfg.model.num_classes,
            dice_weight=self.sadg_cfg.losses.seg.dice_weight,
            ce_weight=self.sadg_cfg.losses.seg.ce_weight,
        )
        return self.sadg_loss

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def get_dataloaders(self):
        """Build multi-domain DataLoaders dynamically managed by nnUNet configuration_manager."""
        batch_size = self.configuration_manager.batch_size
        num_domains = getattr(self.sadg_cfg.model.hdm, 'num_domains', 3)
        samples_per_domain = max(batch_size // num_domains, 1)
        print(f"[SAGD] Building multi-domain DataLoaders (batch_size={batch_size}, {samples_per_domain} samples/domain, num_iterations={self.num_iterations_per_epoch})...", flush=True)

        loaders = get_multi_domain_dataloaders(
            self.sadg_cfg,
            batch_size=batch_size,
            num_iterations=self.num_iterations_per_epoch,
        )

        train_wrapper = MultiDomainDataLoaderWrapper(loaders['train'], self.device)
        val_wrapper = SingleDomainDataLoaderWrapper(loaders['validation'], self.device)

        return train_wrapper, val_wrapper

    def train_step(self, batch) -> dict:
        """
        Multi-domain SAGD training step.
        batch is dict[domain_id] -> {'images': Tensor, 'label': Tensor}
        """
        self.optimizer.zero_grad(set_to_none=True)

        # Clear SAS cache at start of each epoch (handled by iteration count)
        sas_module = getattr(self.network, 'sas', None)
        if sas_module is None and hasattr(self.network, 'backbone'):
            sas_module = getattr(self.network.backbone, 'sas', None)
        if sas_module is None and hasattr(self.network, 'module'):
            sas_module = getattr(self.network.module.backbone, 'sas', None)

        if sas_module is not None:
            if hasattr(self, '_iter_count'):
                self._iter_count += 1
            else:
                self._iter_count = 0

            if self._iter_count % self.num_iterations_per_epoch == 0:
                sas_module.clear_cache()

        # Multi-domain forward via self.network(batch) (ensures DDP registers NCCL autograd hooks)
        output = self.network(batch)

        # Get primary domain labels (slice to match output batch size)
        primary_labels = batch[0]['label'] if 0 in batch else None
        if primary_labels is not None:
            primary_labels = primary_labels[:output['seg_logits'].shape[0]]

        if primary_labels is None:
            # No primary domain in this batch, skip
            return {'loss': 0.0}

        # Compute loss
        loss_dict = self.sadg_loss(
            seg_logits=output['seg_logits'],
            labels=primary_labels,
            primary_tokens=output.get('primary_tokens', None),
            auxiliary_tokens_list=output.get('aux_tokens_list', None),
            auxiliary_logits_list=output.get('aux_seg_list', None),
        )

        l = loss_dict['total']

        # Safe step-skip on NaN/Inf: DO NOT use nan_to_num to mask bad loss,
        # as corrupted gradients from fake loss values destroy all network weights.
        # This was the root cause of Epoch 201 NaN collapse (train_loss stuck at 0.5).
        if torch.isnan(l) or torch.isinf(l):
            self.optimizer.zero_grad(set_to_none=True)
            print(f"[SAGD WARNING] NaN/Inf loss detected at step, skipping backward", flush=True)
            return {'loss': 0.0}

        grad_clip = getattr(self.sadg_cfg.training, 'grad_clip', 12.0)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), grad_clip)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy().item()}

    def validation_step(self, batch: dict) -> dict:
        """Standard single-domain validation on OpenEDS."""
        data = batch['data']  # [B, T, 1, 192, 192]
        target = batch['target']  # [B, 1, 192, 192]
        target_squeezed = target.squeeze(1).long()

        with torch.no_grad():
            output = self.network(data, labels=target_squeezed)

            if isinstance(output, dict):
                seg_logits = output['seg_logits']
            else:
                seg_logits = output

            # Compute segmentation loss only for validation (using cached loss fn)
            l = self.val_loss_fn(seg_logits, target_squeezed)

            num_cls = seg_logits.shape[1]
            axes = [0] + list(range(2, seg_logits.ndim))
            predicted_onehot = F.one_hot(seg_logits.argmax(1), num_cls).permute(0, 3, 1, 2).float()
            target_onehot = F.one_hot(target.squeeze(1).long().clamp(min=0, max=3), num_cls).permute(0, 3, 1, 2).float()

            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_onehot, target_onehot, axes=axes, mask=None)

            tp_hard = tp.detach().cpu().numpy()[1:]
            fp_hard = fp.detach().cpu().numpy()[1:]
            fn_hard = fn.detach().cpu().numpy()[1:]

            dice_dict = compute_dice_score(
                seg_logits, target_squeezed,
                num_classes=seg_logits.shape[1],
            )

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
            'mean_dice': dice_dict['Mean_Dice'],
        }

    def on_validation_epoch_end(self, val_outputs: list):
        """DDP-safe validation epoch aggregation for multi-class dice logging."""
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)
        loss_val_local = np.mean(outputs_collated['loss'])

        if self.is_ddp and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()

            tps = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(tps, tp)
            tp = np.sum(tps, axis=0)

            fps = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(fps, fp)
            fp = np.sum(fps, axis=0)

            fns = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(fns, fn)
            fn = np.sum(fns, axis=0)

            losses_val = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(losses_val, loss_val_local)
            loss_here = np.mean(losses_val)
        else:
            loss_here = loss_val_local

        global_dc_per_class = [2 * i / (2 * i + j + k + 1e-8) for i, j, k in zip(tp, fp, fn)]
        iou_per_class = [i / (i + j + k + 1e-8) for i, j, k in zip(tp, fp, fn)]
        precision_per_class = [i / (i + j + 1e-8) for i, j in zip(tp, fp)]
        recall_per_class = [i / (i + k + 1e-8) for i, k in zip(tp, fn)]

        mean_fg_dice = float(np.nanmean(global_dc_per_class))
        mean_iou = float(np.nanmean(iou_per_class))
        mean_precision = float(np.nanmean(precision_per_class))
        mean_recall = float(np.nanmean(recall_per_class))

        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', float(loss_here), self.current_epoch)

        # Cache extended metrics in trainer for WandB logging
        self._latest_eval_metrics = {
            'iou_per_class': iou_per_class,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'mean_iou': mean_iou,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
        }

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        maybe_mkdir_p(self.output_folder)
        self.set_deep_supervision_enabled(self.enable_deep_supervision)
        self.print_plans()
        empty_cache(self.device)

        if self.local_rank == 0 and HAS_WANDB:
            run_name = f"SAGD_Vivim_fold{self.fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb_mode = os.environ.get('WANDB_MODE', 'online')
            print(f"[SAGD] WandB run: {run_name} (mode={wandb_mode})", flush=True)
            wandb.init(
                project="sagd-eyeball",
                name=run_name,
                mode=wandb_mode,
                config={
                    "trainer": self.__class__.__name__,
                    "fold": self.fold,
                    "temporal_window": self.sadg_cfg.data.temporal_window,
                    "num_epochs": self.num_epochs,
                    "initial_lr": self.initial_lr,
                    "use_sas": self.sadg_cfg.ablation.use_sas,
                    "use_hdm": self.sadg_cfg.ablation.use_hdm,
                    "use_sga": self.sadg_cfg.ablation.use_sga,
                    "num_domains": self.sadg_cfg.model.hdm.num_domains,
                    "input_resolution": list(self.sadg_cfg.data.input_resolution),
                },
            )

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.local_rank == 0 and HAS_WANDB:
            try:
                epoch_idx = self.current_epoch - 1
                train_losses = self.logger.get_value('train_losses', step=None)
                val_losses = self.logger.get_value('val_losses', step=None)
                raw_dice = self.logger.get_value('dice_per_class_or_region', step=None)
                mean_fg_list = self.logger.get_value('mean_fg_dice', step=None)

                train_loss = train_losses[-1] if isinstance(train_losses, list) and len(train_losses) > 0 else 0.0
                val_loss = val_losses[-1] if isinstance(val_losses, list) and len(val_losses) > 0 else 0.0
                pseudo_dice = raw_dice[-1] if isinstance(raw_dice, list) and len(raw_dice) > 0 else [0.0, 0.0, 0.0]

                pupil_dice = float(pseudo_dice[0]) if isinstance(pseudo_dice, (list, np.ndarray)) and len(pseudo_dice) > 0 else 0.0
                iris_dice = float(pseudo_dice[1]) if isinstance(pseudo_dice, (list, np.ndarray)) and len(pseudo_dice) > 1 else 0.0
                sclera_dice = float(pseudo_dice[2]) if isinstance(pseudo_dice, (list, np.ndarray)) and len(pseudo_dice) > 2 else 0.0
                mean_fg_dice = mean_fg_list[-1] if isinstance(mean_fg_list, list) and len(mean_fg_list) > 0 else (pupil_dice + iris_dice + sclera_dice) / 3.0

                ext = getattr(self, '_latest_eval_metrics', {})
                iou = ext.get('iou_per_class', [0.0, 0.0, 0.0])
                prec = ext.get('precision_per_class', [0.0, 0.0, 0.0])
                rec = ext.get('recall_per_class', [0.0, 0.0, 0.0])

                print(f"[SAGD Detailed Metrics] Epoch {epoch_idx} -> train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f} | "
                      f"Dice: [Pupil: {pupil_dice:.4f}, Iris: {iris_dice:.4f}, Sclera: {sclera_dice:.4f}, Mean: {mean_fg_dice:.4f}] | "
                      f"IoU: [Pupil: {iou[0]:.4f}, Iris: {iou[1]:.4f}, Sclera: {iou[2]:.4f}, mIoU: {ext.get('mean_iou', 0.0):.4f}] | "
                      f"Precision: [Pupil: {prec[0]:.4f}, Iris: {prec[1]:.4f}, Sclera: {prec[2]:.4f}, Mean: {ext.get('mean_precision', 0.0):.4f}] | "
                      f"Recall: [Pupil: {rec[0]:.4f}, Iris: {rec[1]:.4f}, Sclera: {rec[2]:.4f}, Mean: {ext.get('mean_recall', 0.0):.4f}]", flush=True)

                wandb.log({
                    "epoch": epoch_idx,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "pupil_dice": pupil_dice,
                    "iris_dice": iris_dice,
                    "sclera_dice": sclera_dice,
                    "mean_fg_dice": mean_fg_dice,
                    "pupil_iou": float(iou[0]),
                    "iris_iou": float(iou[1]),
                    "sclera_iou": float(iou[2]),
                    "mean_iou": float(ext.get('mean_iou', 0.0)),
                    "pupil_precision": float(prec[0]),
                    "iris_precision": float(prec[1]),
                    "sclera_precision": float(prec[2]),
                    "mean_precision": float(ext.get('mean_precision', 0.0)),
                    "pupil_recall": float(rec[0]),
                    "iris_recall": float(rec[1]),
                    "sclera_recall": float(rec[2]),
                    "mean_recall": float(ext.get('mean_recall', 0.0)),
                    "learning_rate": self.lr_scheduler.get_last_lr()[0],
                })
            except Exception as e:
                print(f"[SAGD WandB Error] {e}", flush=True)

    def on_train_end(self):
        super().on_train_end()
        if self.local_rank == 0 and HAS_WANDB:
            wandb.finish()

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        """Build Vivim + SAGD network dynamically managed by nnUNet ConfigurationManager."""
        config_path_candidates = [
            os.path.join(
                os.environ.get('NNUNET_PROJECT_ROOT', ''),
                'configs', 'sadg_vivim.yaml'
            ),
            os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')),
                'configs', 'sadg_vivim.yaml'
            ),
        ]
        cfg = None
        for cp in config_path_candidates:
            if os.path.exists(cp):
                cfg = load_config(cp)
                break
        if cfg is None:
            raise FileNotFoundError(f"Cannot find sadg_vivim.yaml in: {config_path_candidates}")

        arch_info = configuration_manager.configuration.get('architecture', {})
        print(f"[SAGD] Building Vivim + SAS + HDM + SGA network from nnUNet ConfigurationManager (num_classes={num_output_channels}, patch_size={configuration_manager.patch_size}, batch_size={configuration_manager.batch_size})!", flush=True)
        return VivimSAGDWrapper(cfg)
