"""
Visualization utilities for BEV occupancy predictions.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch

from .tools import denormalize_img


def add_ego_vehicle(ax, bx, dx, color="#76b900"):
    """Draw the ego vehicle footprint on a BEV plot."""
    W = 1.85
    pts = np.array([
        [-4.084 / 2.0 + 0.5, W / 2.0],
        [4.084 / 2.0 + 0.5, W / 2.0],
        [4.084 / 2.0 + 0.5, -W / 2.0],
        [-4.084 / 2.0 + 0.5, -W / 2.0],
    ])
    pts = (pts - bx[:2]) / dx[:2]
    pts[:, [0, 1]] = pts[:, [1, 0]]
    ax.fill(pts[:, 0], pts[:, 1], color=color, zorder=5)


def plot_camera_views(imgs, fig=None, axes=None):
    """
    Display all 6 camera images in a 2×3 grid.

    Args:
        imgs: (6, 3, H, W) tensor of normalized images
    """
    cam_names = [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    ]
    if fig is None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 6))

    for idx, ax in enumerate(axes.flat):
        if idx < len(imgs):
            img = denormalize_img(imgs[idx])
            ax.imshow(img)
            ax.set_title(cam_names[idx], fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_bev_occupancy(pred, gt, bx, dx, save_path=None):
    """
    Plot predicted and ground truth BEV occupancy side by side.

    Args:
        pred: (H, W) numpy array — predicted occupancy probabilities
        gt:   (H, W) numpy array — binary ground truth
        bx:   grid center offset
        dx:   grid cell size
        save_path: optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Prediction heatmap
    im0 = axes[0].imshow(
        pred, cmap="RdYlGn_r", vmin=0, vmax=1,
        origin="lower", aspect="equal",
    )
    add_ego_vehicle(axes[0], bx, dx)
    axes[0].set_title("Predicted Occupancy", fontsize=14, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Ground truth
    im1 = axes[1].imshow(
        gt, cmap="RdYlGn_r", vmin=0, vmax=1,
        origin="lower", aspect="equal",
    )
    add_ego_vehicle(axes[1], bx, dx)
    axes[1].set_title("Ground Truth", fontsize=14, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay comparison
    overlay = np.zeros((*pred.shape, 3))
    tp = (pred > 0.5) & (gt > 0.5)
    fp = (pred > 0.5) & (gt <= 0.5)
    fn = (pred <= 0.5) & (gt > 0.5)
    overlay[tp] = [0.0, 0.8, 0.0]   # green = true positive
    overlay[fp] = [0.9, 0.1, 0.1]   # red   = false positive
    overlay[fn] = [0.2, 0.4, 0.9]   # blue  = false negative

    axes[2].imshow(overlay, origin="lower", aspect="equal")
    add_ego_vehicle(axes[2], bx, dx, color="#FFD700")
    axes[2].set_title("Comparison (G=TP, R=FP, B=FN)", fontsize=14, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Y (cells)")
        ax.set_ylabel("X (cells)")

    fig.suptitle("BEV Occupancy — Lift-Splat-Shoot", fontsize=16, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    return fig


def plot_full_result(imgs, pred, gt, bx, dx, save_path=None):
    """
    Combined visualization: camera views + BEV occupancy.

    Args:
        imgs:  (6, 3, H, W) camera images
        pred:  (H, W) predicted occupancy
        gt:    (H, W) ground truth
        bx, dx: grid parameters
        save_path: optional save path
    """
    fig = plt.figure(figsize=(20, 14))

    # Camera views (top 2 rows)
    cam_names = [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    ]
    for idx in range(6):
        ax = fig.add_subplot(3, 3, idx + 1)
        img = denormalize_img(imgs[idx])
        ax.imshow(img)
        ax.set_title(cam_names[idx], fontsize=9)
        ax.axis("off")

    # BEV prediction (bottom left)
    ax_pred = fig.add_subplot(3, 3, 7)
    ax_pred.imshow(pred, cmap="RdYlGn_r", vmin=0, vmax=1, origin="lower")
    add_ego_vehicle(ax_pred, bx, dx)
    ax_pred.set_title("Predicted", fontsize=12, fontweight="bold")

    # BEV ground truth (bottom center)
    ax_gt = fig.add_subplot(3, 3, 8)
    ax_gt.imshow(gt, cmap="RdYlGn_r", vmin=0, vmax=1, origin="lower")
    add_ego_vehicle(ax_gt, bx, dx)
    ax_gt.set_title("Ground Truth", fontsize=12, fontweight="bold")

    # Overlay (bottom right)
    ax_ov = fig.add_subplot(3, 3, 9)
    overlay = np.zeros((*pred.shape, 3))
    tp = (pred > 0.5) & (gt > 0.5)
    fp = (pred > 0.5) & (gt <= 0.5)
    fn = (pred <= 0.5) & (gt > 0.5)
    overlay[tp] = [0, 0.8, 0]
    overlay[fp] = [0.9, 0.1, 0.1]
    overlay[fn] = [0.2, 0.4, 0.9]
    ax_ov.imshow(overlay, origin="lower")
    add_ego_vehicle(ax_ov, bx, dx, color="#FFD700")
    ax_ov.set_title("Comparison", fontsize=12, fontweight="bold")

    fig.suptitle("Lift-Splat-Shoot BEV Occupancy", fontsize=16, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved full result to {save_path}")

    return fig
