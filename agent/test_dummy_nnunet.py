import sys
import torch

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))

try:
    import nnunetv2
    print("nnUNetv2 imported successfully from:", nnunetv2.__file__)
except ImportError as e:
    print("nnUNetv2 import failed:", e)
    sys.exit(1)

print("Environment verification passed!")
