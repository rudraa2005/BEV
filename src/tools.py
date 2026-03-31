"""
Utility functions for the Lift-Splat-Shoot BEV pipeline.
Adapted from NVIDIA LSS (Philion & Fidler, ECCV 2020).
"""

import numpy as np
import torch
import torchvision
from PIL import Image


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def gen_dx_bx(xbound, ybound, zbound):
    """Compute voxel size (dx), lower-left corner (bx), and grid count (nx)."""
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])
    return dx, bx, nx


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def get_rot(h):
    """2D rotation matrix for angle h (radians)."""
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate):
    """Apply augmentations to a PIL image and track the post-transform."""
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # update post-homography
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


# ---------------------------------------------------------------------------
# ImageNet normalisation
# ---------------------------------------------------------------------------

class NormalizeInverse(torchvision.transforms.Normalize):
    """Undo ImageNet normalization for visualization."""
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
))

denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    torchvision.transforms.ToPILImage(),
))


# ---------------------------------------------------------------------------
# Voxel pooling helpers
# ---------------------------------------------------------------------------

def cumsum_trick(x, geom_feats, ranks):
    """Aggregate features in the same voxel using cumulative sums."""
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    """Memory-efficient cumulative sum for voxel pooling with custom backward."""

    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        ctx.save_for_backward(kept)
        ctx.mark_non_differentiable(geom_feats)
        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1
        val = gradx[back]
        return val, None, None


# ---------------------------------------------------------------------------
# Loss & metrics
# ---------------------------------------------------------------------------

class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer(
            "pos_weight", torch.tensor([pos_weight], dtype=torch.float32)
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight
        )

    def forward(self, ypred, ytgt):
        return self.loss_fn(ypred, ytgt)

def get_batch_iou(preds, binimgs):
    """Compute IoU for a batch. Assumes preds have NOT been sigmoided yet."""
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if (union > 0) else 1.0


def get_val_info(model, valloader, loss_fn, device, use_tqdm=False):
    """Run validation and return loss + IoU."""
    from tqdm import tqdm
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    print("Running evaluation...")
    loader = tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            preds = model(
                allimgs.to(device), rots.to(device), trans.to(device),
                intrins.to(device), post_rots.to(device), post_trans.to(device),
            )
            binimgs = binimgs.to(device)
            total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]
            intersect, union, _ = get_batch_iou(preds, binimgs)
            total_intersect += intersect
            total_union += union

    model.train()
    return {
        "loss": total_loss / len(valloader.dataset),
        "iou": total_intersect / total_union if total_union > 0 else 0.0,
    }
