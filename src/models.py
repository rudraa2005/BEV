"""
Lift-Splat-Shoot model for BEV occupancy prediction.
Adapted from NVIDIA LSS (Philion & Fidler, ECCV 2020).

Architecture:
  CamEncode  — EfficientNet-B0 backbone → depth + context features per camera
  Voxel Pool — Lift frustum features to 3D → splat onto BEV grid
  BevEncode  — ResNet-18 BEV decoder → binary occupancy output
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Up(nn.Module):
    """Upsample + concat skip connection + double conv."""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


# ---------------------------------------------------------------------------
# Camera encoder  (per-image feature extraction + depth prediction)
# ---------------------------------------------------------------------------

class CamEncode(nn.Module):
    """
    Extracts features from each camera image and predicts a depth distribution.
    Output: frustum features of shape (B*N, C, D, fH, fW)
    """

    def __init__(self, D, C, downsample):
        super().__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.up1 = Up(320 + 112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_eff_depth(self, x):
        """Run EfficientNet backbone and extract multi-scale features."""
        endpoints = dict()

        # stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
            prev_x = x

        endpoints["reduction_{}".format(len(endpoints) + 1)] = x

        # U-Net style upsampling with skip connection
        x = self.up1(endpoints["reduction_5"], endpoints["reduction_4"])
        return x

    def get_depth_feat(self, x):
        """Extract depth distribution and context features."""
        x = self.get_eff_depth(x)
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, : self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)
        return depth, new_x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)
        return x


# ---------------------------------------------------------------------------
# BEV encoder  (decode BEV feature grid → occupancy)
# ---------------------------------------------------------------------------

class BevEncode(nn.Module):
    """ResNet-18 based decoder that processes the BEV feature grid."""

    def __init__(self, inC, outC):
        super().__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(
            inC, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)
        return x


# ---------------------------------------------------------------------------
# Full Lift-Splat-Shoot pipeline
# ---------------------------------------------------------------------------

class LiftSplatShoot(nn.Module):
    """
    End-to-end model: multi-camera images → BEV occupancy grid.

    1. LIFT   — predict depth distribution per pixel, create frustum features
    2. SPLAT  — project frustum to 3D ego coords, pool into BEV voxels
    3. DECODE — run BEV encoder to produce occupancy predictions
    """

    def __init__(self, grid_conf, data_aug_conf, outC):
        super().__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(
            self.grid_conf["xbound"],
            self.grid_conf["ybound"],
            self.grid_conf["zbound"],
        )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        self.use_quickcumsum = True

    def create_frustum(self):
        """Create a frustum grid in image coordinates × depth."""
        ogfH, ogfW = self.data_aug_conf["final_dim"]
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = (
            torch.arange(*self.grid_conf["dbound"], dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape
        xs = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )
        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """
        Map frustum points to 3D ego-frame coordinates.
        Returns: B x N x D x H x W x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation (augmentation)
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )

        # camera-to-ego: unproject with intrinsics, then rotate + translate
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Run camera encoder on all images. Returns B x N x D x H x W x C."""
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def voxel_pooling(self, geom_feats, x):
        """Pool frustum features into BEV voxels via cumulative sum trick."""
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten
        x = x.reshape(Nprime, C)
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out-of-bounds
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # sort by voxel index for cumsum
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)
            + geom_feats[:, 1] * (self.nx[2] * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # efficient voxel aggregation
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # scatter into dense grid  (B x C x Z x X x Y)
        final = torch.zeros(
            (B, C, int(self.nx[2].item()), int(self.nx[0].item()), int(self.nx[1].item())),
            device=x.device,
        )
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z dimension → (B x C*Z x X x Y)  (Z=1 for our config)
        final = torch.cat(final.unbind(dim=2), 1)
        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        """Full lift-splat pipeline: images → BEV features."""
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)
        x = self.voxel_pooling(geom, x)
        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        """
        Args:
            x:          (B, N, 3, H, W)  camera images
            rots:       (B, N, 3, 3)     camera-to-ego rotation
            trans:      (B, N, 3)        camera-to-ego translation
            intrins:    (B, N, 3, 3)     camera intrinsics
            post_rots:  (B, N, 3, 3)     post-augmentation rotation
            post_trans: (B, N, 3)        post-augmentation translation

        Returns:
            (B, 1, 200, 200)  occupancy logits (apply sigmoid for probability)
        """
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    """Factory function to instantiate the model."""
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
