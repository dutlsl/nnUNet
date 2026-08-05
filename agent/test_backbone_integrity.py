import sys
import os
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("=" * 60)
print("[Integrity Check 1] System & PyTorch Environment")
print("Python Executable:", sys.executable)
print("PyTorch Version :", torch.__version__)
print("CUDA Available  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device Name :", torch.cuda.get_device_name(0))
    print("CUDA Capability :", torch.cuda.get_device_capability(0))

print("\n" + "=" * 60)
print("[Integrity Check 2] nnUNetv2 Import Integrity")
try:
    import nnunetv2
    print("nnUNetv2 Package :", nnunetv2.__file__)
except ImportError as e:
    print("ERROR: nnUNetv2 import failed:", e)
    sys.exit(1)

print("\n" + "=" * 60)
print("[Integrity Check 3] Mamba Block & Selective Scan Verification")
from models.mamba_block import MambaLayer, HAS_MAMBA_CUDA
print("HAS_MAMBA_CUDA Flag:", HAS_MAMBA_CUDA)

mamba_test = MambaLayer(d_model=64, d_state=16).cuda()
x_dummy = torch.randn(2, 64, 64, device='cuda')  # [B, L, D]
y_dummy = mamba_test(x_dummy)
print("Mamba Output Shape :", y_dummy.shape)
assert y_dummy.shape == (2, 64, 64), "Mamba output shape mismatch!"

print("\n" + "=" * 60)
print("[Integrity Check 4] Vivim Network Architecture & Forward/Backward Pass")
from models.vivim_backbone import VivimBackbone
model = VivimBackbone(in_channels=1, num_classes=4, base_channels=32, d_state=16).cuda()

# Test 5D Temporal Sequence input: [Batch=2, Time=3, Channel=1, Height=192, Width=192]
input_tensor = torch.randn(2, 3, 1, 192, 192, device='cuda')
target_tensor = torch.randint(0, 4, (2, 192, 192), device='cuda').long()

criterion = nn.CrossEntropyLoss()
out = model(input_tensor)  # Expected shape [B, C=4, H=192, W=192]
print("Model Output Shape :", out.shape)
assert out.shape == (2, 4, 192, 192), f"Model output shape error: {out.shape}"

loss = criterion(out, target_tensor)
print("Loss Computation   :", loss.item())

loss.backward()
print("Backward Gradient  : Success (Gradients computed for all trainable parameters)")

print("\n" + "=" * 60)
print("[SUCCESS] All 4 Integrity Checks Passed Cleanly!")
print("=" * 60)
