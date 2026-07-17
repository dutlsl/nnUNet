"""
nnUNetTrainer_ImageNetPretrained

Custom nnUNet trainer that partially transfers ImageNet-pretrained ResNet-34
encoder weights into nnUNet's 2D encoder via shape-matching.

Strategy:
  1. Build the standard nnUNet 2d network (PlainConvUNet or ResidualEncoderUNet).
  2. Load torchvision ResNet-34 with ImageNet pretrained weights.
  3. Iterate over all Conv2d layers in nnUNet's encoder and ResNet, collecting
     weights grouped by shape.
  4. For each shape that appears in both models, transfer ResNet conv weights
     to the corresponding nnUNet encoder conv layers (in order).
  5. Lower initial LR to 1e-3 for fine-tuning stability.

This provides meaningful initialization for encoder stages where channel
dimensions coincidentally match ResNet (typically from stage 1 / 64ch onward).
"""

import inspect
import warnings
from collections import defaultdict

import torch
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


class nnUNetTrainer_ImageNetPretrained(nnUNetTrainer):
    """nnUNet trainer with partial ImageNet pretrained encoder initialization."""

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Lower LR for fine-tuning (pretrained encoder needs gentler updates)
        self.initial_lr = 1e-3
        self.wandb_initialized = False

    @staticmethod
    def _get_encoder_module(network):
        """Extract the encoder module from the network, handling DDP/compiled wrappers."""
        mod = network
        if isinstance(mod, DDP):
            mod = mod.module
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod.encoder

    @staticmethod
    def _collect_conv_weights_by_shape(module: nn.Module):
        """
        Walk a module and collect all Conv2d weight tensors, grouped by shape.
        Returns: dict[shape_tuple] -> list of (name, parameter)
        """
        result = defaultdict(list)
        for name, m in module.named_modules():
            if isinstance(m, nn.Conv2d):
                shape = tuple(m.weight.shape)
                result[shape].append((name, m.weight))
        return result

    def _transfer_imagenet_weights(self, network):
        """
        Load ResNet-34 ImageNet pretrained weights and transfer to nnUNet encoder
        where conv weight shapes match.
        """
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        except ImportError:
            self.print_to_log_file(
                "WARNING: torchvision not available. Skipping ImageNet weight transfer. "
                "Training from scratch."
            )
            return 0

        encoder = self._get_encoder_module(network)

        # Collect conv weights by shape from both models
        nnunet_convs = self._collect_conv_weights_by_shape(encoder)
        resnet_convs = self._collect_conv_weights_by_shape(resnet)

        # Find matching shapes
        matching_shapes = set(nnunet_convs.keys()) & set(resnet_convs.keys())

        transferred_count = 0
        transferred_details = []

        for shape in matching_shapes:
            nn_layers = nnunet_convs[shape]
            rn_layers = resnet_convs[shape]

            # Transfer min(len(nn), len(rn)) layers, in order
            num_to_transfer = min(len(nn_layers), len(rn_layers))
            for i in range(num_to_transfer):
                nn_name, nn_param = nn_layers[i]
                rn_name, rn_param = rn_layers[i]

                with torch.no_grad():
                    nn_param.copy_(rn_param.data)

                transferred_count += 1
                transferred_details.append(
                    f"  {rn_name} ({list(shape)}) -> encoder.{nn_name}"
                )

        # Log results
        self.print_to_log_file(
            f"\n{'='*60}\n"
            f"ImageNet Pretrained Weight Transfer Summary\n"
            f"{'='*60}\n"
            f"ResNet-34 total Conv2d layers: {sum(len(v) for v in resnet_convs.values())}\n"
            f"nnUNet encoder Conv2d layers: {sum(len(v) for v in nnunet_convs.values())}\n"
            f"Matching shape groups: {len(matching_shapes)}\n"
            f"Weights transferred: {transferred_count}\n"
        )
        if transferred_details:
            self.print_to_log_file("Transfer details:")
            for detail in transferred_details:
                self.print_to_log_file(detail)
        self.print_to_log_file(f"{'='*60}\n")

        return transferred_count

    def initialize(self):
        if not self.was_initialized:
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            # Build the network architecture (standard nnUNet way)
            sig = inspect.signature(self.build_network_architecture)
            if 'plans_manager' in sig.parameters:
                self.network = self.build_network_architecture(
                    self.plans_manager,
                    self.configuration_manager,
                    self.num_input_channels,
                    self.label_manager.num_segmentation_heads,
                    self.enable_deep_supervision
                ).to(self.device)
            else:
                warnings.warn(
                    f"Trainer {self.__class__.__name__} uses the old "
                    "build_network_architecture signature.",
                    DeprecationWarning, stacklevel=2,
                )
                self.network = self.build_network_architecture(
                    self.configuration_manager.network_arch_class_name,
                    self.configuration_manager.network_arch_init_kwargs,
                    self.configuration_manager.network_arch_init_kwargs_req_import,
                    self.num_input_channels,
                    self.label_manager.num_segmentation_heads,
                    self.enable_deep_supervision
                ).to(self.device)

            # === CORE: Transfer ImageNet pretrained weights ===
            num_transferred = self._transfer_imagenet_weights(self.network)
            self.print_to_log_file(
                f"ImageNet pretrained encoder initialization: "
                f"{num_transferred} conv layers transferred."
            )

            # Compile network (if applicable)
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            self.was_initialized = True

            # Initialize W&B logging (only on rank 0)
            if self.local_rank == 0:
                try:
                    import wandb
                    wandb.init(
                        project="nnUNet_OpenEDS2019",
                        name=f"{self.__class__.__name__}__fold_{self.fold}",
                        config={
                            "initial_lr": self.initial_lr,
                            "weight_decay": self.weight_decay,
                            "oversample_foreground_percent": self.oversample_foreground_percent,
                            "num_epochs": self.num_epochs,
                            "batch_size": self.configuration_manager.batch_size,
                            "patch_size": self.configuration_manager.patch_size,
                            "pretrained_encoder": "ResNet-34 (ImageNet pretrained, shape-matching transfer)"
                        }
                    )
                    self.wandb_initialized = True
                    self.print_to_log_file("Weights & Biases logging initialized successfully.")
                except Exception as e:
                    self.print_to_log_file(f"W&B Init failed with error: {e}. Proceeding without W&B.")

            logger_config_hparas = {
                "initial_lr": self.initial_lr,
                "weight_decay": self.weight_decay,
                "oversample_foreground_percent": self.oversample_foreground_percent,
                "probabilistic_oversampling": self.probabilistic_oversampling,
                "num_iterations_per_epoch": self.num_iterations_per_epoch,
                "num_val_iterations_per_epoch": self.num_val_iterations_per_epoch,
                "num_epochs": self.num_epochs,
                "enable_deep_supervision": self.enable_deep_supervision,
                "batch_size": self.configuration_manager.batch_size,
                "pretrained_encoder": "ResNet-34 ImageNet (partial shape-matching transfer)",
            }
            self.logger.update_config({"hparas": logger_config_hparas})
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was "
                "already initialized. That should not happen."
            )

    def on_epoch_end(self):
        # Call base implementation to print to file and handle checkpoints
        super().on_epoch_end()

        # Log to W&B if initialized (on local_rank == 0)
        if self.local_rank == 0 and self.wandb_initialized:
            try:
                import wandb
                # Since super().on_epoch_end() increments self.current_epoch, we look at step = self.current_epoch - 1
                metric_step = self.current_epoch - 1
                
                train_loss = self.logger.get_value('train_losses', step=metric_step)
                val_loss = self.logger.get_value('val_losses', step=metric_step)
                dice_per_class = self.logger.get_value('dice_per_class_or_region', step=metric_step)
                ema_fg_dice = self.logger.get_value('ema_fg_dice', step=metric_step)
                current_lr = self.optimizer.param_groups[0]['lr']

                log_dict = {
                    "epoch": metric_step,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/ema_fg_dice": ema_fg_dice,
                    "lr": current_lr,
                }

                # Log individual classes (1: sclera, 2: iris, 3: pupil)
                class_names = ["sclera", "iris", "pupil"]
                for i, score in enumerate(dice_per_class):
                    if i < len(class_names):
                        log_dict[f"val/dice_{class_names[i]}"] = score
                    else:
                        log_dict[f"val/dice_class_{i}"] = score

                wandb.log(log_dict, step=metric_step)
            except Exception as e:
                self.print_to_log_file(f"W&B Log failed on epoch end: {e}")
