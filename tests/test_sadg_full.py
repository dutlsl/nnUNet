"""
SAGD Full Pipeline Smoke Test.
Verifies all modules (SAS, HDM, SGA, Loss, Backbone) work with dummy input.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

from utils.config import load_config
from models.vivim_backbone import VivimBackbone
from models.sadg_serialization import StructureAwareSerializer, CDS2D, GCS2D
from models.sadg_hdm import HDM2D, IntraDomainMamba, InterDomainFusion
from models.sadg_sga import SpectralGraphAlignment
from losses.sadg_loss import SAGDLoss, DiceCELoss

def test_cds2d():
    print("=== CDS2D ===")
    cds = CDS2D(feature_h=24, feature_w=24, num_neighbors=8)
    feat = torch.randn(2, 256, 24, 24)
    order = cds(feat)
    print(f"  Input: {feat.shape} -> Order: {order.shape}")
    assert order.shape == (2, 576), f"Expected (2, 576), got {order.shape}"
    print("  ✓ CDS2D passed")

def test_gcs2d():
    print("=== GCS2D ===")
    # Use smaller feature map for faster test
    gcs = GCS2D(feature_h=6, feature_w=6, diffusion_time=2.0, num_eigenvectors=8)
    feat = torch.randn(2, 256, 6, 6)
    order = gcs(feat)
    print(f"  Input: {feat.shape} -> Order: {order.shape}")
    assert order.shape == (2, 36), f"Expected (2, 36), got {order.shape}"
    print("  ✓ GCS2D passed")

def test_sas():
    print("=== StructureAwareSerializer ===")
    cfg = load_config(os.path.join(PROJECT_ROOT, 'configs', 'sadg_vivim.yaml'))
    # Override to small for test
    cfg['model']['sas']['feature_h'] = 6
    cfg['model']['sas']['feature_w'] = 6
    cfg['model']['sas']['num_tokens'] = 36

    sas = StructureAwareSerializer(cfg)
    feat = torch.randn(2, 256, 6, 6)
    fwd_cds, rev_cds, fwd_gcs, rev_gcs, inv_cds_order = sas(feat)
    print(f"  fwd_cds: {fwd_cds.shape}, rev_gcs: {rev_gcs.shape}, inv_cds_order: {inv_cds_order.shape}")
    assert fwd_cds.shape == (2, 36, 256)
    assert inv_cds_order.shape == (2, 36)
    print("  ✓ SAS passed")

def test_ism():
    print("=== IntraDomainMamba ===")
    ism = IntraDomainMamba(d_model=256, d_state=16, d_conv=4, expand=2).to(DEVICE)
    tokens = torch.randn(2, 36, 256, device=DEVICE)
    out = ism(tokens)
    print(f"  Input: {tokens.shape} -> Output: {out.shape}")
    assert out.shape == tokens.shape
    print("  ✓ ISM passed")

def test_irf():
    print("=== InterDomainFusion ===")
    irf = InterDomainFusion(d_model=256, num_domains=3, chunk_size=16).to(DEVICE)
    tokens_list = [
        torch.randn(2, 36, 256, device=DEVICE),
        torch.randn(2, 36, 256, device=DEVICE),
        torch.randn(2, 36, 256, device=DEVICE),
    ]
    out_list = irf(tokens_list)
    print(f"  Input: 3 x {tokens_list[0].shape} -> Output: {len(out_list)} domains")
    for i, out in enumerate(out_list):
        print(f"    Domain {i}: {out.shape}")
    print("  ✓ IRF passed")

def test_hdm():
    print("=== HDM2D ===")
    cfg = load_config(os.path.join(PROJECT_ROOT, 'configs', 'sadg_vivim.yaml'))
    hdm = HDM2D(cfg).to(DEVICE)
    tokens_list = [
        torch.randn(2, 36, 256, device=DEVICE),
        torch.randn(2, 36, 256, device=DEVICE),
        torch.randn(2, 36, 256, device=DEVICE),
    ]
    hdm.train()
    out = hdm(tokens_list, domain_ids=[0, 1, 2])
    print(f"  Training output: {out.shape}")
    assert out.shape == (2, 36, 256)

    hdm.eval()
    out_eval = hdm(tokens_list, domain_ids=[0, 1, 2])
    print(f"  Eval output: {out_eval.shape}")
    print("  ✓ HDM passed")

def test_sga():
    print("=== SpectralGraphAlignment ===")
    sga = SpectralGraphAlignment(
        num_classes=4, feature_dim=256,
        num_prototypes_per_class=4, spectral_k=8,
    ).to(DEVICE)
    tokens = torch.randn(2, 36, 256, device=DEVICE)
    labels = torch.randint(0, 4, (2, 6, 6), device=DEVICE)

    sga.train()
    out_train = sga(tokens, labels)
    print(f"  Training: {tokens.shape} -> {out_train.shape}")
    assert out_train.shape == tokens.shape

    sga.eval()
    out_eval = sga(tokens)
    print(f"  Eval: {tokens.shape} -> {out_eval.shape}")
    assert out_eval.shape == tokens.shape
    print("  ✓ SGA passed")

def test_loss():
    print("=== SAGDLoss ===")
    cfg = load_config(os.path.join(PROJECT_ROOT, 'configs', 'sadg_vivim.yaml'))
    loss_fn = SAGDLoss(cfg).to(DEVICE)

    seg_logits = torch.randn(2, 4, 192, 192, device=DEVICE)
    labels = torch.randint(0, 4, (2, 192, 192), device=DEVICE)
    primary_tokens = torch.randn(2, 36, 256, device=DEVICE)
    aux_tokens = [torch.randn(2, 36, 256, device=DEVICE)]
    aux_logits = [torch.randn(2, 4, 192, 192, device=DEVICE)]

    loss_dict = loss_fn(seg_logits, labels, primary_tokens, aux_tokens, aux_logits)
    print(f"  total: {loss_dict['total'].item():.4f}")
    print(f"  seg: {loss_dict['seg'].item():.4f}")
    print(f"  domain_consistency: {loss_dict['domain_consistency'].item():.4f}")
    print(f"  structural_contrastive: {loss_dict['structural_contrastive'].item():.4f}")
    print("  ✓ SAGDLoss passed")

def test_backbone_sagd():
    print("=== VivimBackbone + SAGD (full pipeline) ===")
    cfg = load_config(os.path.join(PROJECT_ROOT, 'configs', 'sadg_vivim.yaml'))

    backbone = VivimBackbone(
        in_channels=1, num_classes=4, base_channels=32,
        d_state=16, d_conv=4, expand=2,
        use_mamba=True, use_sadg=True,
    ).to(DEVICE)

    # Attach SAS (with reduced size for test)
    cfg['model']['sas']['feature_h'] = 24
    cfg['model']['sas']['feature_w'] = 24
    sas = StructureAwareSerializer(cfg).to(DEVICE)
    backbone.sas = sas

    # Attach SGA
    sga = SpectralGraphAlignment(
        num_classes=4, feature_dim=256,
        num_prototypes_per_class=4, spectral_k=8,
    ).to(DEVICE)
    backbone.sga = sga

    # Forward pass
    x = torch.randn(1, 3, 1, 192, 192, device=DEVICE)  # [B=1, T=3, C=1, H=192, W=192]
    labels = torch.randint(0, 4, (1, 192, 192), device=DEVICE)

    backbone.train()
    output = backbone(x, labels=labels)

    assert isinstance(output, dict), f"Expected dict, got {type(output)}"
    print(f"  seg_logits: {output['seg_logits'].shape}")
    print(f"  serialized_tokens: {output['serialized_tokens'].shape}")
    print(f"  bottleneck_features: {output['bottleneck_features'].shape}")

    assert output['seg_logits'].shape == (1, 4, 192, 192)
    print("  ✓ Full backbone pipeline passed")

def test_backbone_domain_forward():
    print("=== VivimBackbone.forward_domain + decode_from_tokens ===")
    cfg = load_config(os.path.join(PROJECT_ROOT, 'configs', 'sadg_vivim.yaml'))

    backbone = VivimBackbone(
        in_channels=1, num_classes=4, base_channels=32,
        d_state=16, d_conv=4, expand=2,
        use_mamba=True, use_sadg=True,
    ).to(DEVICE)

    cfg['model']['sas']['feature_h'] = 24
    cfg['model']['sas']['feature_w'] = 24
    sas = StructureAwareSerializer(cfg).to(DEVICE)
    backbone.sas = sas

    x = torch.randn(1, 3, 1, 192, 192, device=DEVICE)
    out = backbone.forward_domain(x, domain_id=0)

    print(f"  bottleneck: {out['bottleneck'].shape}")
    print(f"  serialized_tokens: {out['serialized_tokens'].shape}")
    print(f"  spatial_shape: {out['spatial_shape']}")

    seg = backbone.decode_from_tokens(
        out['serialized_tokens'], out['skips'], out['spatial_shape'],
        inv_cds_order=out['inv_cds_order']
    )
    print(f"  decoded seg_logits: {seg.shape}")
    assert seg.shape == (1, 4, 192, 192)
    print("  ✓ Domain forward + decode passed")


if __name__ == '__main__':
    print("=" * 60)
    print("SAGD Full Pipeline Smoke Test")
    print("=" * 60)

    test_cds2d()
    test_gcs2d()
    test_sas()
    test_ism()
    test_irf()
    test_hdm()
    test_sga()
    test_loss()
    test_backbone_sagd()
    test_backbone_domain_forward()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
