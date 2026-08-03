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
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime

# Ensure project root is in sys.path
# File is at: nnunetv2/training/nnUNetTrainer/variants/network_architecture/
# 5 levels up -> nnUNet project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
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
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from models.vivim_backbone import VivimBackbone
from models.sadg_serialization import StructureAwareSerializer
from models.sadg_hdm import HDM2D
from models.sadg_sga import SpectralGraphAlignment
from losses.sadg_loss import SAGDLoss
from datasets.multi_domain_dataset import (
    get_multi_domain_dataloaders,
    collate_multi_domain,
    RITnetPreprocessor,
    OpenEDSDomainDataset,
)
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
        """Standard forward: single domain inference."""
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

        # HDM: hierarchical domain modeling
        if self.hdm is not None and len(serialized_list) > 1 and self.training:
            hdm_output = self.hdm(serialized_list, domain_ids)

            # SGA: update prototypes with HDM-fused tokens
            if self.sga is not None and 0 in domain_batches:
                primary_labels = domain_batches[0]['label']
                hdm_output = self.sga(hdm_output, primary_labels)

            # Decode primary domain from HDM-fused tokens
            primary_out = domain_outputs[0]
            primary_seg = self.backbone.decode_from_tokens(
                hdm_output, primary_out['skips'], primary_out['spatial_shape']
            )
        elif 0 in domain_outputs:
            # Single domain or no HDM
            primary_out = domain_outputs[0]
            tokens = primary_out.get('serialized_tokens', None)

            if tokens is not None:
                if self.sga is not None:
                    tokens = self.sga(tokens, domain_batches[0].get('label', None))
                primary_seg = self.backbone.decode_from_tokens(
                    tokens, primary_out['skips'], primary_out['spatial_shape']
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

        # Also decode auxiliary domains if available (for domain consistency loss)
        aux_seg_list = []
        for d_id in sorted(domain_batches.keys()):
            if d_id == 0:
                continue
            if d_id in domain_outputs:
                aux_out = domain_outputs[d_id]
                if self.hdm is not None and len(serialized_list) > 1:
                    # Use ISM-only output for auxiliary (we already got HDM for primary)
                    idx = domain_ids.index(d_id) if d_id in domain_ids else None
                    if idx is not None and idx < len(serialized_list):
                        aux_tokens = self.hdm.ism_blocks[d_id](serialized_list[idx])
                        aux_seg = self.backbone.decode_from_tokens(
                            aux_tokens, aux_out['skips'], aux_out['spatial_shape']
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
        self.initial_lr = 1e-3
        self.weight_decay = 1e-4

        # Load SAGD config
        config_path = os.path.join(PROJECT_ROOT, 'configs', 'sadg_vivim.yaml')
        self.sadg_cfg = load_config(config_path)

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
        """AdamW with PolyLR scheduler."""
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

        # Try to load pretrained weights from baseline Vivim
        ckpt_paths = [
            os.path.join(PROJECT_ROOT, 'nnUNet_results', 'Dataset600_OpenEDS2019',
                         'nnUNetTrainer_Vivim__nnUNetPlans__2d', 'fold_0', 'checkpoint_final.pth'),
            '/home/iulab0/PycharmProjects/nnUNet/nnUNet_results/vivim_weakmed_400x400/checkpoint_best.pth',
        ]

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
                    # Skip WeakMed/Sphere related weights
                    if any(skip in k for skip in ['sphere_head', 'weakmed', 'eyeball_head']):
                        continue
                    # Try direct match
                    if k in model_keys:
                        new_state_dict[k] = v
                    # Try backbone prefix
                    elif f"backbone.{k}" in model_keys:
                        new_state_dict[f"backbone.{k}"] = v

                if new_state_dict:
                    missing, unexpected = self.network.load_state_dict(
                        new_state_dict, strict=False
                    )
                    print(f"[SAGD] Loaded {len(new_state_dict)} weights. Missing: {len(missing)}", flush=True)
                    break

    def _build_loss(self):
        """Build SAGD combined loss."""
        self.sadg_loss = SAGDLoss(self.sadg_cfg)
        return self.sadg_loss

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def get_dataloaders(self):
        """Build multi-domain DataLoaders from SAGD config."""
        print("[SAGD] Building multi-domain DataLoaders (OpenEDS + Swirski + LPW)...", flush=True)

        loaders = get_multi_domain_dataloaders(self.sadg_cfg)

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
        if hasattr(self.network, 'sas') and self.network.sas is not None:
            if hasattr(self, '_iter_count'):
                self._iter_count += 1
            else:
                self._iter_count = 0

            if self._iter_count % self.num_iterations_per_epoch == 0:
                self.network.sas.clear_cache()

        # Multi-domain forward
        if hasattr(self.network, 'module'):
            output = self.network.module.forward_multi_domain(batch)
        else:
            output = self.network.forward_multi_domain(batch)

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

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """Standard single-domain validation on OpenEDS."""
        data = batch['data']  # [B, T, 1, 192, 192]
        target = batch['target']  # [B, 1, 192, 192]
        target_squeezed = target.squeeze(1).long()

        with torch.no_grad():
            if hasattr(self.network, 'module'):
                output = self.network.module(data, labels=target_squeezed)
            else:
                output = self.network(data, labels=target_squeezed)

            if isinstance(output, dict):
                seg_logits = output['seg_logits']
            else:
                seg_logits = output

            # Compute segmentation loss only for validation
            from losses.sadg_loss import DiceCELoss
            val_loss_fn = DiceCELoss(num_classes=seg_logits.shape[1])
            l = val_loss_fn(seg_logits, target_squeezed)

            axes = [0] + list(range(2, seg_logits.ndim))
            output_seg = seg_logits.argmax(1)[:, None]
            predicted_onehot = torch.zeros(
                seg_logits.shape, device=seg_logits.device, dtype=torch.float16
            )
            predicted_onehot.scatter_(1, output_seg, 1)

            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_onehot, target, axes=axes, mask=None)

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
            print(f"[SAGD] WandB run: {run_name}", flush=True)
            wandb.init(
                project="sagd-eyeball",
                name=run_name,
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
                train_loss = self.logger.get_value('train_losses', step=-1)
                val_loss = self.logger.get_value('val_losses', step=-1)
                mean_fg_dice = self.logger.get_value('mean_fg_dice', step=-1)

                wandb.log({
                    "epoch": epoch_idx,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "mean_fg_dice": mean_fg_dice,
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
        """Build Vivim + SAGD network from config."""
        config_path = os.path.join(PROJECT_ROOT, 'configs', 'sadg_vivim.yaml')
        cfg = load_config(config_path)

        print("[SAGD] Building Vivim + SAS + HDM + SGA network!", flush=True)
        return VivimSAGDWrapper(cfg)
