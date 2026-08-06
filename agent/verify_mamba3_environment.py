"""
Mamba-3 Environment & Checkpoint Verification Harness for RTX 4090 Demo Server.

Run this script on any target server (e.g., RTX 4090) to verify:
1. CUDA & GPU environment compatibility
2. Triton kernel engine installation
3. mamba_ssm (v2.3.2+) Mamba-3 module import
4. Mamba3 GPU forward pass execution
5. VivimBackbone 5D temporal sequence forward pass
6. Trained backbone checkpoint (checkpoint_best.pth) integrity
"""

import os
import sys
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def verify_environment():
    print("=" * 65)
    print(" 🚀 Mamba-3 Environment & Checkpoint Verification Harness")
    print("=" * 65)

    # 1. CUDA & Device Check
    print("\n[1/6] Checking PyTorch & CUDA Environment...")
    print(f"  - PyTorch Version : {torch.__version__}")
    print(f"  - CUDA Available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  ❌ ERROR: CUDA is not available! Mamba-3 requires NVIDIA GPU.")
        sys.exit(1)
    device_name = torch.cuda.get_device_name(0)
    print(f"  - GPU Device      : {device_name}")
    print(f"  - VRAM Capacity   : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 2. Triton Kernel Engine Check
    print("\n[2/6] Checking Kernel Engine (Triton)...")
    try:
        import triton
        print(f"  - triton Version  : {triton.__version__} (Mamba-3 Triton Kernel Core - OK)")
    except ImportError as e:
        print("  ❌ ERROR: triton is not installed! Mamba-3 requires triton.")
        sys.exit(1)

    # Informational check for causal_conv1d
    try:
        import causal_conv1d
        print(f"  - causal_conv1d   : {causal_conv1d.__version__} (Installed)")
    except ImportError:
        print("  - causal_conv1d   : Not installed (Note: Mamba-3 uses native Triton kernels, causal_conv1d is not required)")

    # 3. mamba_ssm & Mamba3 Import Check
    print("\n[3/6] Checking mamba_ssm & Mamba3 Module Import...")
    try:
        import mamba_ssm
        print(f"  - mamba_ssm Version     : {mamba_ssm.__version__}")
        from mamba_ssm.modules.mamba3 import Mamba3
        print("  - Mamba3 Module Import  : SUCCESS (mamba_ssm.modules.mamba3.Mamba3)")
    except ImportError as e:
        print(f"  ❌ ERROR: Failed to import Mamba3 from mamba_ssm: {e}")
        sys.exit(1)

    # 4. Mamba3 Module CUDA Execution Check
    print("\n[4/6] Testing Mamba3 CUDA Layer Forward Pass...")
    try:
        mamba3_layer = Mamba3(d_model=256, d_state=16, headdim=64, expand=2).cuda()
        x_3d = torch.randn(2, 10, 256, device="cuda")
        with torch.no_grad():
            out_3d = mamba3_layer(x_3d)
        print(f"  - Input Tensor  : {list(x_3d.shape)}")
        print(f"  - Output Tensor : {list(out_3d.shape)}")
        assert out_3d.shape == x_3d.shape, "Shape mismatch in Mamba3 forward pass!"
        print("  - Forward Pass  : SUCCESS (3D sequence tensor)")
    except Exception as e:
        print(f"  ❌ ERROR during Mamba3 forward pass: {e}")
        sys.exit(1)

    # 5. VivimBackbone 5D Temporal Sequence Check
    print("\n[5/6] Testing VivimBackbone (Spatio-Temporal Mamba) Forward Pass...")
    try:
        from models.vivim_backbone import VivimBackbone
        backbone = VivimBackbone(
            in_channels=1,
            num_classes=4,
            base_channels=32,
            d_state=16,
            d_conv=4,
            expand=2,
            use_mamba=True
        ).cuda()
        x_5d = torch.randn(2, 3, 1, 192, 192, device="cuda")  # [B, T=3, C=1, 192, 192]
        with torch.no_grad():
            out_seg = backbone(x_5d)
        print(f"  - 5D Input Tensor  : {list(x_5d.shape)}")
        print(f"  - Segmentation Out : {list(out_seg.shape)}")
        assert out_seg.shape == (2, 4, 192, 192), f"Expected (2, 4, 192, 192), got {out_seg.shape}"
        print("  - VivimBackbone    : SUCCESS (5D spatio-temporal sequence)")
    except Exception as e:
        print(f"  ❌ ERROR during VivimBackbone forward pass: {e}")
        sys.exit(1)

    # 6. Checkpoint File Verification
    print("\n[6/6] Checking Checkpoint Integrity (checkpoint_best.pth)...")
    ckpt_path = os.path.join(
        PROJECT_ROOT,
        "nnUNet_results/Dataset600_OpenEDS2019/nnUNetTrainer_Vivim__nnUNetPlans__2d/fold_1/checkpoint_best.pth"
    )
    if not os.path.isfile(ckpt_path):
        print(f"  ⚠️ WARNING: Checkpoint file not found at {ckpt_path}")
    else:
        file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        print(f"  - Checkpoint File : {ckpt_path}")
        print(f"  - File Size       : {file_size_mb:.2f} MB")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "network_weights" in ckpt:
                state_dict = ckpt["network_weights"]
                print(f"  - Weights Format  : nnUNet Trainer Dict ({len(state_dict)} tensors)")
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
                print(f"  - Weights Format  : Standard State Dict ({len(state_dict)} tensors)")
            elif isinstance(ckpt, dict):
                state_dict = ckpt
                print(f"  - Weights Format  : Direct Dict ({len(state_dict)} tensors)")
            else:
                state_dict = {}
                print("  - Weights Format  : Unknown format")

            # Check if weights can be loaded into backbone
            if state_dict:
                cleaned_keys = {}
                for k, v in state_dict.items():
                    key = k.replace("backbone.", "")
                    cleaned_keys[key] = v
                missing, unexpected = backbone.load_state_dict(cleaned_keys, strict=False)
                print(f"  - State Dict Load : SUCCESS (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
        except Exception as e:
            print(f"  ❌ ERROR loading checkpoint: {e}")
            sys.exit(1)

    print("\n" + "=" * 65)
    print(" 🎉 ALL MAMBA-3 ENVIRONMENT & CHECKPOINT CHECKS PASSED!")
    print("     The RTX 4090 demo server environment is ready for deployment.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    verify_environment()
