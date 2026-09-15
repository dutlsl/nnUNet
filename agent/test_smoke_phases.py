import os, sys, torch

PROJECT_ROOT = '/home/iulab1/PycharmProjects/nnUNet'
sys.path.insert(0, PROJECT_ROOT)

if not hasattr(torch, 'float8_e8m0fnu'):
    torch.float8_e8m0fnu = None
if not hasattr(torch, 'float4_e2m1fn_x2'):
    torch.float4_e2m1fn_x2 = None

from utils.config import load_config
from datasets.dual_stream_dataset import get_dual_stream_dataloaders
from datasets.boundary_window_dataset import BoundaryWindowDataset, DataLoader
from models.vivim_backbone import VivimBackbone
from models.vivim_backbone_multiframe import VivimBackboneMultiFrame
from losses.sparse_label_loss import SparseLabelLoss
from losses.multiframe_loss import MultiFrameMaskedLoss

print("=== 1. Testing Phase 1 on GPU 0 ===")
cfg1 = load_config(os.path.join(PROJECT_ROOT, 'configs/phase1_sparse_t5.yaml'))
loaders1 = get_dual_stream_dataloaders(cfg1)
batch1 = next(iter(loaders1['train']))
print(f"Phase 1 batch loaded: images={batch1['images'].shape}, label={batch1['label'].shape}, has_label={batch1['has_label']}")

m1 = cfg1.model.mamba
model1 = VivimBackbone(
    in_channels=1, num_classes=cfg1.model.num_classes,
    base_channels=cfg1.model.base_channels,
    d_state=m1.d_state, d_conv=m1.d_conv, expand=m1.expand,
    use_mamba=cfg1.ablation.use_mamba,
    mamba_version=getattr(cfg1.ablation, 'mamba_version', 'mamba3'),
    mamba_headdim=getattr(m1, 'headdim', 64),
).to('cuda:0')

criterion1 = SparseLabelLoss(num_classes=cfg1.model.num_classes).to('cuda:0')
with torch.cuda.amp.autocast():
    out1 = model1(batch1['images'].to('cuda:0'))
    loss1, sup1, unsup1 = criterion1(out1, batch1['label'].to('cuda:0'), batch1['has_label'].to('cuda:0'), cfg=cfg1, epoch=1)
loss1.backward()
print(f"Phase 1 SUCCESS! Total loss: {loss1.item():.4f} (sup: {sup1.item():.4f}, unsup: {unsup1.item():.4f})")

print("\n=== 2. Testing Phase 2 on GPU 1 ===")
cfg2 = load_config(os.path.join(PROJECT_ROOT, 'configs/phase2_multiframe_t5.yaml'))
sem_root = cfg2.data.semantic_gt_root
seq_root = cfg2.data.sequence_root
train_ds2 = BoundaryWindowDataset(
    os.path.join(sem_root, 'train', 'images'),
    os.path.join(sem_root, 'train', 'labels'),
    os.path.join(seq_root, 'train'),
    temporal_window=cfg2.data.temporal_window,
    crop=tuple(cfg2.data.input_crop),
    padded_resolution=tuple(cfg2.data.target_resolution),
)
loader2 = DataLoader(train_ds2, batch_size=cfg2.training.batch_size, shuffle=False)
batch2 = next(iter(loader2))
print(f"Phase 2 batch loaded: images={batch2['images'].shape}, labels={batch2['labels'].shape}, mask={batch2['label_mask'].shape}")

m2 = cfg2.model.mamba
model2 = VivimBackboneMultiFrame(
    in_channels=1, num_classes=cfg2.model.num_classes,
    base_channels=cfg2.model.base_channels,
    d_state=m2.d_state, d_conv=m2.d_conv, expand=m2.expand,
    use_mamba=cfg2.ablation.use_mamba,
    mamba_version=getattr(cfg2.ablation, 'mamba_version', 'mamba3'),
    mamba_headdim=getattr(m2, 'headdim', 64),
).to('cuda:1')

criterion2 = MultiFrameMaskedLoss(num_classes=cfg2.model.num_classes).to('cuda:1')
with torch.cuda.amp.autocast():
    out2 = model2(batch2['images'].to('cuda:1'), output_all_frames=True)
    loss2, sup2, smt2 = criterion2(out2, batch2['labels'].to('cuda:1'), batch2['label_mask'].to('cuda:1'), cfg=cfg2, epoch=1)
loss2.backward()
print(f"Phase 2 SUCCESS! Total loss: {loss2.item():.4f} (sup: {sup2.item():.4f}, smt: {smt2.item():.4f})")

# Test inference mode of Phase 2
with torch.no_grad():
    infer_out = model2(batch2['images'].to('cuda:1'), output_all_frames=False)
print(f"Phase 2 Inference mode output: {infer_out.shape} (Expected: [B, C, H, W])")
print("\n>>> ALL SMOKE TESTS PASSED PERFECTLY! <<<")
