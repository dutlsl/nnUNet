"""
Environment & Mamba3 Engine Verification Harness Script
Tests:
1. PyTorch & CUDA GPU device availability
2. mamba_ssm CUDA selective_scan_fn kernel import & execution
3. MambaLayer & TemporalMambaBlock forward/backward gradient check on GPU
4. VivimBackbone 5D tensor input [B=2, T=3, C=1, H=448, W=448] -> [2, 4, 448, 448] output check
5. nnUNetv2 & nnUNetTrainer_Vivim instantiation check
"""

import sys
import torch
import torch.nn as nn

print("==================================================")
print(" 🚀 1. System Environment Check")
print("==================================================")
print(f"Python Version : {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ ERROR: CUDA is not available!")
    sys.exit(1)

device = torch.device("cuda:0")
print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")

print("\n==================================================")
print(" ⚡ 2. Mamba SSM & CUDA Kernel Import Check")
print("==================================================")
try:
    import mamba_ssm
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from models.mamba_block import HAS_MAMBA_CUDA
    print(f"✅ mamba_ssm version : {getattr(mamba_ssm, '__version__', 'Installed')}")
    print(f"✅ HAS_MAMBA_CUDA flag: {HAS_MAMBA_CUDA}")
    if not HAS_MAMBA_CUDA:
        print("⚠️ WARNING: mamba_ssm CUDA kernel is using PyTorch fallback!")
    else:
        print("🎉 CUDA selective_scan_fn kernel is ACTIVE!")
except Exception as e:
    print(f"❌ mamba_ssm CUDA kernel import failed: {e}")
    sys.exit(1)

print("\n==================================================")
print(" 🧪 3. MambaLayer & TemporalMambaBlock Dummy Tensor Test")
print("==================================================")
from models.mamba_block import MambaLayer
from models.temporal_mamba import TemporalMambaBlock

# Test MambaLayer directly on GPU
try:
    mamba_layer = MambaLayer(d_model=256, d_state=16, d_conv=4, expand=2).to(device)
    dummy_mamba_in = torch.randn(4, 64, 256, device=device, requires_grad=True)  # [B, L, D]
    mamba_out = mamba_layer(dummy_mamba_in)
    loss = mamba_out.sum()
    loss.backward()
    print(f"✅ MambaLayer Forward Output Shape : {list(mamba_out.shape)}")
    print(f"✅ MambaLayer Backward Gradient Check: OK (grad norm = {dummy_mamba_in.grad.norm().item():.4f})")
except Exception as e:
    print(f"❌ MambaLayer Execution Failed: {e}")
    sys.exit(1)

# Test TemporalMambaBlock on 5D Temporal sequence
try:
    tmb = TemporalMambaBlock(channels=256, d_state=16, d_conv=4, expand=2).to(device)
    dummy_tmb_in = torch.randn(2, 3, 256, 56, 56, device=device, requires_grad=True)  # [B, T, C, H, W]
    tmb_out = tmb(dummy_tmb_in)
    loss_tmb = tmb_out.sum()
    loss_tmb.backward()
    print(f"✅ TemporalMambaBlock Output Shape  : {list(tmb_out.shape)}")
    print(f"✅ TemporalMambaBlock Backward Check: OK (grad norm = {dummy_tmb_in.grad.norm().item():.4f})")
except Exception as e:
    print(f"❌ TemporalMambaBlock Execution Failed: {e}")
    sys.exit(1)

print("\n==================================================")
print(" 🧠 4. VivimBackbone 5D Sequence Dummy Output Check")
print("==================================================")
from models.vivim_backbone import VivimBackbone

try:
    sphere_cfg = {
        'hidden_dim': 128,
        'max_radius': 200.0,
        'min_radius': 30.0,
        'sharpness': 20.0,
    }
    model = VivimBackbone(
        in_channels=1,
        num_classes=4,
        base_channels=32,
        d_state=16,
        d_conv=4,
        expand=2,
        use_mamba=True,
        use_sphere_head=True,
        sphere_head_cfg=sphere_cfg,
    ).to(device)

    # OpenEDS sequence dummy input: [B=2, T=3, C=1, H=448, W=448]
    dummy_video = torch.randn(2, 3, 1, 448, 448, device=device, requires_grad=True)
    out_dict = model(dummy_video)

    print(f"✅ Vivim seg_logits Shape  : {list(out_dict['seg_logits'].shape)}")
    print(f"✅ Vivim sphere_mask Shape : {list(out_dict['sphere_mask'].shape)}")
    print(f"✅ Vivim sphere_params     : {out_dict['sphere_params'].detach().cpu().numpy()}")

    # NaN / Inf Check
    has_nan = torch.isnan(out_dict['seg_logits']).any().item()
    has_inf = torch.isinf(out_dict['seg_logits']).any().item()
    if has_nan or has_inf:
        print(f"❌ Output contains invalid values! NaN: {has_nan}, Inf: {has_inf}")
        sys.exit(1)
    else:
        print("✅ Output values clean! No NaN / No Inf.")

    # Backward Pass Check
    dummy_target = torch.randint(0, 4, (2, 448, 448), device=device)
    ce_loss = nn.CrossEntropyLoss()(out_dict['seg_logits'], dummy_target)
    ce_loss.backward()
    print(f"✅ Vivim Full Model Backward Check: OK (Loss = {ce_loss.item():.4f})")

except Exception as e:
    print(f"❌ VivimBackbone Execution Failed: {e}")
    sys.exit(1)

print("\n==================================================")
print(" 📦 5. nnUNetv2 Framework & Custom Trainer Import Check")
print("==================================================")
try:
    import nnunetv2
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_Vivim import VivimBackboneNNUNet
    print(f"✅ nnUNetv2 Installed Path : {nnunetv2.__file__}")
    
    wrapper = VivimBackboneNNUNet().to(device)
    dummy_wrapper_in = torch.randn(2, 3, 1, 448, 448, device=device)
    wrapper_out = wrapper(dummy_wrapper_in)
    print(f"✅ VivimBackboneNNUNet Wrapper Output Keys: {list(wrapper_out.keys())}")
except Exception as e:
    print(f"❌ nnUNetv2 Integration Check Failed: {e}")
    sys.exit(1)

print("\n==================================================")
print(" 🎉 ALL ENVIRONMENT & MAMBA3 DUMMY INPUT VERIFICATIONS PASSED SUCCESSFULLY! 🎉")
print("==================================================")
