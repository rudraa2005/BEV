"""
Training script for Lift-Splat-Shoot BEV occupancy prediction.

Usage:
    python train.py                             # defaults (RTX 4050 friendly)
    python train.py --dataroot /path/to/nuscenes --bsz 4
    python train.py --gpuid -1                  # CPU only (slow)
"""

import argparse
import os
import numpy as np
import torch
from time import time

from src.models import compile_model
from src.data import compile_data
from src.tools import SimpleLoss, get_batch_iou, get_val_info


def parse_args():
    p = argparse.ArgumentParser(description="Train Lift-Splat-Shoot BEV model")

    # dataset
    p.add_argument("--version", default="mini", help="nuscenes version: mini | trainval")
    p.add_argument("--dataroot", default=r"C:\Users\Dhruvi\Desktop\v1.0-mini",
                    help="Path to the nuScenes root directory")

    # training
    p.add_argument("--nepochs", type=int, default=50, help="Number of epochs")
    p.add_argument("--gpuid", type=int, default=0, help="GPU ID (-1 for CPU)")
    p.add_argument("--bsz", type=int, default=2, help="Batch size")
    p.add_argument("--nworkers", type=int, default=2, help="Dataloader workers")
    p.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-7)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--pos-weight", type=float, default=5.0,
                    help="Positive class weight for BCE loss")

    # image
    p.add_argument("--H", type=int, default=900)
    p.add_argument("--W", type=int, default=1600)
    p.add_argument("--final-dim", nargs=2, type=int, default=[128, 352])
    p.add_argument("--resize-lim", nargs=2, type=float, default=[0.15, 0.3])
    p.add_argument("--bot-pct-lim", nargs=2, type=float, default=[0.0, 0.22])
    p.add_argument("--rot-lim", nargs=2, type=float, default=[-10.0, 10.0])
    p.add_argument("--rand-flip", action="store_true", default=True)
    p.add_argument("--ncams", type=int, default=6)

    # grid
    p.add_argument("--xbound", nargs=3, type=float, default=[-50.0, 50.0, 0.5])
    p.add_argument("--ybound", nargs=3, type=float, default=[-50.0, 50.0, 0.5])
    p.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    p.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

    # output
    p.add_argument("--logdir", default="./runs", help="Tensorboard log directory")

    return p.parse_args()


def train():
    args = parse_args()

    # ---- configs ----
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
        "H": args.H,
        "W": args.W,
        "rand_flip": args.rand_flip,
        "bot_pct_lim": args.bot_pct_lim,
        "cams": [
            "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
        ],
        "Ncams": args.ncams,
    }

    # ---- data ----
    print("=" * 60)
    print("Loading nuScenes dataset...")
    print("=" * 60)
    trainloader, valloader = compile_data(
        args.version, args.dataroot,
        data_aug_conf=data_aug_conf, grid_conf=grid_conf,
        bsz=args.bsz, nworkers=args.nworkers,
        parser_name="segmentationdata",
    )

    # ---- device ----
    if args.gpuid < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpuid}")
    print(f"\nUsing device: {device}")

    # ---- model ----
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable:,} trainable\n")

    # ---- optimizer & loss ----
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = SimpleLoss(args.pos_weight).to(device)

    # ---- tensorboard ----
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.logdir)
        use_tb = True
        print(f"TensorBoard logs: {args.logdir}")
    except ImportError:
        print("TensorBoard not available, logging to console only.")
        use_tb = False

    # ---- training loop ----
    val_step = 1000 if args.version == "mini" else 10000
    os.makedirs(args.logdir, exist_ok=True)

    model.train()
    counter = 0
    best_iou = 0.0

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    for epoch in range(args.nepochs):
        np.random.seed()
        epoch_loss = 0.0
        epoch_steps = 0

        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()

            preds = model(
                imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
            )
            binimgs = binimgs.to(device)

            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()

            counter += 1
            epoch_steps += 1
            epoch_loss += loss.item()
            t1 = time()

            if counter % 10 == 0:
                print(
                    f"[Epoch {epoch+1}/{args.nepochs}] "
                    f"Step {counter} | Loss: {loss.item():.4f} | "
                    f"Time: {t1 - t0:.2f}s"
                )
                if use_tb:
                    writer.add_scalar("train/loss", loss.item(), counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                print(f"  → Train IoU: {iou:.4f}")
                if use_tb:
                    writer.add_scalar("train/iou", iou, counter)
                    writer.add_scalar("train/epoch", epoch, counter)
                    writer.add_scalar("train/step_time", t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, device, use_tqdm=True)
                print(f"\n{'='*40}")
                print(f"VALIDATION @ step {counter}")
                print(f"  Loss: {val_info['loss']:.4f}")
                print(f"  IoU:  {val_info['iou']:.4f}")
                print(f"{'='*40}\n")

                if use_tb:
                    writer.add_scalar("val/loss", val_info["loss"], counter)
                    writer.add_scalar("val/iou", val_info["iou"], counter)

                # save checkpoint
                mname = os.path.join(args.logdir, f"model{counter}.pt")
                print(f"Saving checkpoint: {mname}")
                torch.save(model.state_dict(), mname)

                if val_info["iou"] > best_iou:
                    best_iou = val_info["iou"]
                    best_path = os.path.join(args.logdir, "model_best.pt")
                    torch.save(model.state_dict(), best_path)
                    print(f"  ★ New best IoU: {best_iou:.4f} → saved to {best_path}")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\n--- Epoch {epoch+1}/{args.nepochs} complete | Avg loss: {avg_loss:.4f} ---\n")

    # ---- final save ----
    final_path = os.path.join(args.logdir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    print(f"Best validation IoU: {best_iou:.4f}")

    if use_tb:
        writer.close()


if __name__ == "__main__":
    train()
