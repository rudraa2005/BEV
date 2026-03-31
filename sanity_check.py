"""
Quick sanity check: verify data loading + model forward pass works correctly.
"""
import torch
import sys

print("=" * 60)
print("SANITY CHECK: Lift-Splat-Shoot BEV Pipeline")
print("=" * 60)

# 1. Check imports
print("\n[1/5] Checking imports...")
try:
    from src.models import compile_model
    from src.data import compile_data
    from src.tools import gen_dx_bx, SimpleLoss, get_batch_iou
    print("  [OK] All imports successful")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

# 2. Check CUDA
print("\n[2/5] Checking CUDA...")
if torch.cuda.is_available():
    print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  [OK] VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    device = torch.device("cuda:0")
else:
    print("  [WARN] CUDA not available, using CPU")
    device = torch.device("cpu")

# 3. Check data loading
print("\n[3/5] Loading nuScenes mini dataset...")
grid_conf = {
    "xbound": [-50.0, 50.0, 0.5],
    "ybound": [-50.0, 50.0, 0.5],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [4.0, 45.0, 1.0],
}
data_aug_conf = {
    "resize_lim": (0.193, 0.225),
    "final_dim": (128, 352),
    "rot_lim": (-5.4, 5.4),
    "H": 900, "W": 1600,
    "rand_flip": True,
    "bot_pct_lim": (0.0, 0.22),
    "cams": [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    ],
    "Ncams": 6,
}

try:
    trainloader, valloader = compile_data(
        "mini", r"C:\Users\Dhruvi\Desktop\v1.0-mini",
        data_aug_conf=data_aug_conf, grid_conf=grid_conf,
        bsz=1, nworkers=0,
    )
    print(f"  [OK] Train samples: {len(trainloader.dataset)}")
    print(f"  [OK] Val samples:   {len(valloader.dataset)}")
except Exception as e:
    print(f"  [FAIL] Data loading failed: {e}")
    sys.exit(1)

# 4. Check one batch
print("\n[4/5] Loading one batch...")
try:
    batch = next(iter(trainloader))
    imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
    print(f"  [OK] Images:    {imgs.shape}")
    print(f"  [OK] Rots:      {rots.shape}")
    print(f"  [OK] Trans:     {trans.shape}")
    print(f"  [OK] Intrins:   {intrins.shape}")
    print(f"  [OK] BinImgs:   {binimgs.shape}")
    print(f"  [OK] GT range:  [{binimgs.min():.0f}, {binimgs.max():.0f}]")
except Exception as e:
    print(f"  [FAIL] Batch loading failed: {e}")
    sys.exit(1)

# 5. Check forward pass
print("\n[5/5] Running model forward pass...")
try:
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] Model parameters: {total_params:,}")

    with torch.no_grad():
        preds = model(
            imgs.to(device), rots.to(device), trans.to(device),
            intrins.to(device), post_rots.to(device), post_trans.to(device),
        )
    print(f"  [OK] Output shape: {preds.shape}")
    print(f"  [OK] Output range: [{preds.min().item():.4f}, {preds.max().item():.4f}]")

    # check IoU computation
    _, _, iou = get_batch_iou(preds, binimgs.to(device))
    print(f"  [OK] Initial IoU (untrained): {iou:.4f}")

    # check loss computation
    loss_fn = SimpleLoss(5.0).to(device)
    loss = loss_fn(preds, binimgs.to(device))
    print(f"  [OK] Initial loss: {loss.item():.4f}")

except Exception as e:
    print(f"  [FAIL] Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL CHECKS PASSED! Pipeline is ready for training.")
print("=" * 60)
print("\nNext step: python train.py")
