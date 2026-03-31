"""
Evaluation and visualization script for the trained LSS BEV model.

Usage:
    python evaluate.py --modelf runs/model_best.pt
    python evaluate.py --modelf runs/model_best.pt --compare-geometric
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import compile_model
from src.data import compile_data
from src.tools import gen_dx_bx, get_batch_iou, SimpleLoss, denormalize_img
from src.visualize import plot_bev_occupancy, plot_full_result, add_ego_vehicle


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained LSS BEV model")
    p.add_argument("--modelf", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--dataroot", default=r"C:\Users\Dhruvi\Desktop\v1.0-mini")
    p.add_argument("--version", default="mini")
    p.add_argument("--gpuid", type=int, default=0)
    p.add_argument("--bsz", type=int, default=1)
    p.add_argument("--nworkers", type=int, default=2)
    p.add_argument("--outdir", default="./eval_results", help="Output directory")
    p.add_argument("--num-vis", type=int, default=10,
                    help="Number of samples to visualize")
    p.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold for prediction binarization")
    p.add_argument("--compare-geometric", action="store_true",
                    help="Also run geometric BEV for comparison")

    # match training config
    p.add_argument("--H", type=int, default=900)
    p.add_argument("--W", type=int, default=1600)
    p.add_argument("--final-dim", nargs=2, type=int, default=[128, 352])
    p.add_argument("--resize-lim", nargs=2, type=float, default=[0.193, 0.225])
    p.add_argument("--bot-pct-lim", nargs=2, type=float, default=[0.0, 0.22])
    p.add_argument("--rot-lim", nargs=2, type=float, default=[-5.4, 5.4])
    p.add_argument("--ncams", type=int, default=6)
    p.add_argument("--xbound", nargs=3, type=float, default=[-50.0, 50.0, 0.5])
    p.add_argument("--ybound", nargs=3, type=float, default=[-50.0, 50.0, 0.5])
    p.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    p.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

    return p.parse_args()


def compute_distance_weighted_error(pred, gt, bx, dx):
    """
    Distance-weighted error: errors closer to ego are penalized more heavily.
    Returns scalar error value.
    """
    H, W = pred.shape
    # compute distance from center of grid
    y_coords = np.arange(H) * dx[0] + bx[0] - dx[0] / 2
    x_coords = np.arange(W) * dx[1] + bx[1] - dx[1] / 2
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    dist = np.sqrt(xx**2 + yy**2)

    # weight: closer = higher weight (inverse distance, capped)
    weight = 1.0 / np.maximum(dist, 1.0)
    weight /= weight.sum()

    error = np.abs(pred - gt)
    weighted_error = (error * weight).sum()
    return float(weighted_error)


def evaluate():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---- config ----
    grid_conf = {
        "xbound": args.xbound,
        "ybound": args.ybound,
        "zbound": args.zbound,
        "dbound": args.dbound,
    }
    data_aug_conf = {
        "resize_lim": args.resize_lim,
        "final_dim": tuple(args.final_dim),
        "rot_lim": args.rot_lim,
        "H": args.H, "W": args.W,
        "rand_flip": False,
        "bot_pct_lim": args.bot_pct_lim,
        "cams": [
            "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
        ],
        "Ncams": args.ncams,
    }

    dx, bx, nx = gen_dx_bx(
        grid_conf["xbound"], grid_conf["ybound"], grid_conf["zbound"]
    )
    dx_np, bx_np = dx.numpy(), bx.numpy()

    # ---- device ----
    device = torch.device("cpu") if args.gpuid < 0 else torch.device(f"cuda:{args.gpuid}")
    print(f"Using device: {device}")

    # ---- load data ----
    print("Loading validation data...")
    _, valloader = compile_data(
        args.version, args.dataroot,
        data_aug_conf=data_aug_conf, grid_conf=grid_conf,
        bsz=args.bsz, nworkers=args.nworkers,
    )

    # ---- load model ----
    print(f"Loading model from {args.modelf}...")
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.load_state_dict(torch.load(args.modelf, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!\n")

    # ---- evaluate ----
    total_intersect = 0.0
    total_union = 0.0
    total_dwe = 0.0
    num_samples = 0
    vis_count = 0

    print("Running evaluation on validation set...")
    print("-" * 50)

    with torch.no_grad():
        for batch_idx, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(valloader):
            preds = model(
                imgs.to(device), rots.to(device), trans.to(device),
                intrins.to(device), post_rots.to(device), post_trans.to(device),
            )
            binimgs_dev = binimgs.to(device)

            # IoU
            intersect, union, iou = get_batch_iou(preds, binimgs_dev)
            total_intersect += intersect
            total_union += union

            # per-sample metrics + visualization
            B = preds.shape[0]
            for i in range(B):
                import cv2
                pred_prob = torch.sigmoid(preds[i, 0]).cpu().numpy()
                gt = binimgs[i, 0].numpy()

                # Clean up predictions (threshold + morphological blur)
                pred_bin = (pred_prob > args.threshold).astype(np.uint8)
                pred_clean = cv2.medianBlur(pred_bin, 3).astype(float)

                # distance-weighted error
                dwe = compute_distance_weighted_error(
                    pred_clean, gt, bx_np, dx_np
                )
                total_dwe += dwe
                num_samples += 1

                # save visualizations
                if vis_count < args.num_vis:
                    vis_count += 1
                    save_path = os.path.join(args.outdir, f"sample_{vis_count:03d}.png")
                    plot_full_result(
                        imgs[i], pred_clean, gt, bx_np, dx_np,
                        save_path=save_path,
                    )
                    plt.close("all")

                    # also save just the BEV comparison
                    bev_path = os.path.join(args.outdir, f"bev_{vis_count:03d}.png")
                    plot_bev_occupancy(pred_clean, gt, bx_np, dx_np, save_path=bev_path)
                    plt.close("all")

            print(f"  Batch {batch_idx+1}/{len(valloader)} | IoU: {iou:.4f}")

    # ---- summary ----
    overall_iou = total_intersect / total_union if total_union > 0 else 0.0
    avg_dwe = total_dwe / max(num_samples, 1)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Samples evaluated:     {num_samples}")
    print(f"  Occupancy IoU:         {overall_iou:.4f}")
    print(f"  Avg Distance-Wt Error: {avg_dwe:.6f}")
    print(f"  Visualizations saved:  {args.outdir}/")
    print("=" * 50)

    # save metrics to JSON
    metrics = {
        "model": args.modelf,
        "num_samples": num_samples,
        "occupancy_iou": round(overall_iou, 4),
        "distance_weighted_error": round(avg_dwe, 6),
    }
    metrics_path = os.path.join(args.outdir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    evaluate()
