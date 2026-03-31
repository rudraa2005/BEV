"""
nuScenes dataset loader for BEV occupancy prediction.
Adapted from NVIDIA LSS (Philion & Fidler, ECCV 2020).
"""

import os
import numpy as np
import torch
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
import torchvision

from .tools import gen_dx_bx, img_transform, normalize_img


class NuscData(torch.utils.data.Dataset):
    """Base nuScenes dataset for BEV tasks."""

    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(
            grid_conf["xbound"], grid_conf["ybound"], grid_conf["zbound"]
        )
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()
        print(self)

    def fix_nuscenes_formatting(self):
        """Handle different nuscenes directory structures."""
        rec = self.ixes[0]
        sampimg = self.nusc.get("sample_data", rec["data"]["CAM_FRONT"])
        imgname = os.path.join(self.nusc.dataroot, sampimg["filename"])

        if not os.path.isfile(imgname):
            print("Adjusting nuscenes file paths...")
            from glob import glob

            def find_name(f):
                d, fi = os.path.split(f)
                d, di = os.path.split(d)
                d, d0 = os.path.split(d)
                d, d1 = os.path.split(d)
                d, d2 = os.path.split(d)
                return di, fi, f"{d2}/{d1}/{d0}/{di}/{fi}"

            fs = glob(os.path.join(self.nusc.dataroot, "samples/*/samples/CAM*/*.jpg"))
            fs += glob(os.path.join(self.nusc.dataroot, "samples/*/samples/LIDAR_TOP/*.pcd.bin"))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f"samples/{di}/{fi}"] = fname
            fs = glob(os.path.join(self.nusc.dataroot, "sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin"))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f"sweeps/{di}/{fi}"] = fname
            for rec in self.nusc.sample_data:
                if rec["channel"] == "LIDAR_TOP" or (
                    rec["is_key_frame"]
                    and rec["channel"] in self.data_aug_conf["cams"]
                ):
                    if rec["filename"] in info:
                        rec["filename"] = info[rec["filename"]]

    def get_scenes(self):
        """Get scene names for this split."""
        split = {
            "v1.0-trainval": {True: "train", False: "val"},
            "v1.0-mini": {True: "mini_train", False: "mini_val"},
        }[self.nusc.version][self.is_train]
        return create_splits_scenes()[split]

    def prepro(self):
        """Filter and sort samples by scene and timestamp."""
        samples = [
            samp
            for samp in self.nusc.sample
            if self.nusc.get("scene", samp["scene_token"])["name"] in self.scenes
        ]
        samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))
        return samples

    def sample_augmentation(self):
        """Compute augmentation parameters (resize, crop, flip, rotate)."""
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]

        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        """Load and augment all camera images for a sample."""
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        for cam in cams:
            samp = self.nusc.get("sample_data", rec["data"][cam])
            imgname = os.path.join(self.nusc.dataroot, samp["filename"])
            img = Image.open(imgname)
            if self.is_train:
                img = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
            intrin = torch.Tensor(sens["camera_intrinsic"])
            rot = torch.Tensor(Quaternion(sens["rotation"]).rotation_matrix)
            tran = torch.Tensor(sens["translation"])

            # apply augmentation
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(
                img, post_rot, post_tran,
                resize=resize, resize_dims=resize_dims,
                crop=crop, flip=flip, rotate=rotate,
            )

            # expand to 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (
            torch.stack(imgs),
            torch.stack(rots),
            torch.stack(trans),
            torch.stack(intrins),
            torch.stack(post_rots),
            torch.stack(post_trans),
        )

    def get_binimg(self, rec):
        """Render binary occupancy ground truth from vehicle annotations."""
        egopose = self.nusc.get(
            "ego_pose",
            self.nusc.get("sample_data", rec["data"]["LIDAR_TOP"])["ego_pose_token"],
        )
        trans = -np.array(egopose["translation"])
        rot = Quaternion(egopose["rotation"]).inverse

        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec["anns"]:
            inst = self.nusc.get("sample_annotation", tok)
            # only vehicles count as "occupied"
            if not inst["category_name"].split(".")[0] == "vehicle":
                continue
            box = Box(inst["translation"], inst["size"], Quaternion(inst["rotation"]))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.0) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):
        """Select camera subset (random during training, all during val)."""
        if self.is_train and self.data_aug_conf["Ncams"] < len(self.data_aug_conf["cams"]):
            cams = np.random.choice(
                self.data_aug_conf["cams"],
                self.data_aug_conf["Ncams"],
                replace=False,
            )
        else:
            cams = self.data_aug_conf["cams"]
        return cams

    def __str__(self):
        return (
            f'NuscData: {len(self)} samples. '
            f'Split: {"train" if self.is_train else "val"}.'
        )

    def __len__(self):
        return len(self.ixes)


class SegmentationData(NuscData):
    """Dataset that returns images + binary occupancy ground truth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec)
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz, nworkers,
                 parser_name="segmentationdata"):
    """Build train and val dataloaders."""
    nusc = NuScenes(
        version="v1.0-{}".format(version),
        dataroot=dataroot,
        verbose=True,
    )
    parser = {
        "segmentationdata": SegmentationData,
    }[parser_name]

    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf, grid_conf=grid_conf)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf, grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(
        traindata,
        batch_size=bsz,
        shuffle=True,
        num_workers=nworkers,
        drop_last=True,
        worker_init_fn=worker_rnd_init,
    )
    valloader = torch.utils.data.DataLoader(
        valdata,
        batch_size=bsz,
        shuffle=False,
        num_workers=nworkers,
    )
    return trainloader, valloader
