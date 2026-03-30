import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def quaternion_to_rotation_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_json(root, table_name):
    with (root / f"{table_name}.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_indexes(meta_root):
    sample = load_json(meta_root, "sample")
    sample_data = load_json(meta_root, "sample_data")
    calibrated_sensor = load_json(meta_root, "calibrated_sensor")
    ego_pose = load_json(meta_root, "ego_pose")
    sensor = load_json(meta_root, "sensor")

    return {
        "sample_order": [row["token"] for row in sample],
        "sample_data": sample_data,
        "calibrated_sensor": {row["token"]: row for row in calibrated_sensor},
        "ego_pose": {row["token"]: row for row in ego_pose},
        "sensor": {row["token"]: row for row in sensor},
    }


def build_sample_bundle(indexes, sample_index=0, sample_token=None):
    if sample_token is None:
        sample_token = indexes["sample_order"][sample_index]

    bundle = {}
    for entry in indexes["sample_data"]:
        if entry["sample_token"] != sample_token or not entry["is_key_frame"]:
            continue

        calib = indexes["calibrated_sensor"][entry["calibrated_sensor_token"]]
        sensor = indexes["sensor"][calib["sensor_token"]]
        bundle[sensor["channel"]] = {
            "data": entry,
            "calib": calib,
            "ego_pose": indexes["ego_pose"][entry["ego_pose_token"]],
        }

    missing = [channel for channel in CAMERA_CHANNELS + ("LIDAR_TOP",) if channel not in bundle]
    if missing:
        raise ValueError(f"Sample {sample_token} is missing required sensors: {missing}")

    return sample_token, bundle


def ego_to_global(points, pose):
    rotation = quaternion_to_rotation_matrix(pose["rotation"])
    translation = np.array(pose["translation"], dtype=np.float64)
    return points @ rotation.T + translation


def global_to_ego(points, pose):
    rotation = quaternion_to_rotation_matrix(pose["rotation"])
    translation = np.array(pose["translation"], dtype=np.float64)
    return (points - translation) @ rotation


def ego_to_sensor(points, calib):
    rotation = quaternion_to_rotation_matrix(calib["rotation"])
    translation = np.array(calib["translation"], dtype=np.float64)
    return (points - translation) @ rotation


def sensor_to_ego(points, calib):
    rotation = quaternion_to_rotation_matrix(calib["rotation"])
    translation = np.array(calib["translation"], dtype=np.float64)
    return points @ rotation.T + translation


def grid_shape(x_limits, y_limits, resolution):
    height = int(round((x_limits[1] - x_limits[0]) / resolution))
    width = int(round((y_limits[1] - y_limits[0]) / resolution))
    return height, width


def build_ground_grid(x_limits, y_limits, resolution):
    height, width = grid_shape(x_limits, y_limits, resolution)

    rows = np.arange(height, dtype=np.float32)
    cols = np.arange(width, dtype=np.float32)

    x_coords = x_limits[1] - (rows + 0.5) * resolution
    y_coords = y_limits[1] - (cols + 0.5) * resolution

    xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
    points = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
    return points, xx, yy


def world_to_grid(xy, x_limits, y_limits, resolution):
    x = xy[..., 0]
    y = xy[..., 1]
    row = np.floor((x_limits[1] - x) / resolution).astype(int)
    col = np.floor((y_limits[1] - y) / resolution).astype(int)
    return row, col


def build_vehicle_mask(xx, yy):
    return (xx >= -2.4) & (xx <= 2.4) & (yy >= -1.1) & (yy <= 1.1)


def build_camera_bev(dataset_root, bundle, x_limits, y_limits, resolution):
    lidar_pose = bundle["LIDAR_TOP"]["ego_pose"]
    ground_points, xx, yy = build_ground_grid(x_limits, y_limits, resolution)
    global_points = ego_to_global(ground_points.reshape(-1, 3), lidar_pose).reshape(ground_points.shape)

    height, width = ground_points.shape[:2]
    bev_sum = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for channel in CAMERA_CHANNELS:
        record = bundle[channel]
        image_path = dataset_root / record["data"]["filename"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load camera image: {image_path}")

        ego_points_at_camera_time = global_to_ego(
            global_points.reshape(-1, 3), record["ego_pose"]
        ).reshape(ground_points.shape)
        camera_points = ego_to_sensor(
            ego_points_at_camera_time.reshape(-1, 3), record["calib"]
        ).reshape(ground_points.shape)

        intrinsics = np.array(record["calib"]["camera_intrinsic"], dtype=np.float64)
        uvw = camera_points @ intrinsics.T
        uv = uvw[..., :2] / np.maximum(uvw[..., 2:], 1e-6)

        sampled = cv2.remap(
            image,
            uv[..., 0].astype(np.float32),
            uv[..., 1].astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        depth = np.linalg.norm(camera_points, axis=-1)
        z = camera_points[..., 2]
        view_alignment = np.clip(z / np.maximum(depth, 1e-6), 0.0, 1.0)

        valid = (
            (z > 1.0)
            & (uv[..., 0] >= 0.0)
            & (uv[..., 0] < image.shape[1] - 1.0)
            & (uv[..., 1] >= image.shape[0] * 0.47)
            & (uv[..., 1] < image.shape[0] - 1.0)
        )

        weight = (view_alignment**4) * np.exp(-0.04 * np.maximum(depth - 8.0, 0.0))
        weight *= valid.astype(np.float32)

        bev_sum += sampled.astype(np.float32) * weight[..., None]
        weight_sum += weight

    bev = np.where(
        weight_sum[..., None] > 0.0,
        bev_sum / np.maximum(weight_sum[..., None], 1e-6),
        0.0,
    ).astype(np.uint8)

    vehicle_mask = build_vehicle_mask(xx, yy)
    bev[vehicle_mask] = (40, 40, 40)

    return bev, weight_sum, vehicle_mask


def interpolate_circular_bins(values):
    valid_bins = np.flatnonzero(values > 0.0)
    if valid_bins.size == 0:
        return values

    extended_bins = np.concatenate([valid_bins - len(values), valid_bins, valid_bins + len(values)])
    extended_values = np.tile(values[valid_bins], 3)

    interpolated = np.interp(np.arange(len(values)), extended_bins, extended_values)
    kernel = np.ones(15, dtype=np.float32) / 15.0
    padded = np.concatenate([interpolated[-7:], interpolated, interpolated[:7]])
    return np.convolve(padded, kernel, mode="valid")


def build_occupancy_grid(dataset_root, bundle, x_limits, y_limits, resolution, observed_bins=720):
    lidar_record = bundle["LIDAR_TOP"]
    lidar_path = dataset_root / lidar_record["data"]["filename"]

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    points = sensor_to_ego(points, lidar_record["calib"])

    valid = (
        (points[:, 0] >= x_limits[0])
        & (points[:, 0] <= x_limits[1])
        & (points[:, 1] >= y_limits[0])
        & (points[:, 1] <= y_limits[1])
        & (points[:, 2] >= -2.5)
        & (points[:, 2] <= 3.0)
    )
    points = points[valid]

    height, width = grid_shape(x_limits, y_limits, resolution)
    observed = np.zeros((height, width), dtype=np.uint8)
    occupied = np.zeros((height, width), dtype=np.uint8)

    angles = np.arctan2(points[:, 1], points[:, 0])
    distances = np.linalg.norm(points[:, :2], axis=1)
    rows, cols = world_to_grid(points[:, :2], x_limits, y_limits, resolution)

    bins = np.linspace(-np.pi, np.pi, observed_bins + 1)
    indices = np.clip(np.digitize(angles, bins) - 1, 0, observed_bins - 1)

    max_distance = np.zeros(observed_bins, dtype=np.float32)
    for index in range(observed_bins):
        mask = indices == index
        if np.any(mask):
            max_distance[index] = distances[mask].max()

    max_distance = interpolate_circular_bins(max_distance)

    polygon = []
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    for angle, distance in zip(bin_centers, max_distance):
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        row, col = world_to_grid(np.array([x, y]), x_limits, y_limits, resolution)
        if 0 <= row < height and 0 <= col < width:
            polygon.append([int(col), int(row)])

    if len(polygon) > 2:
        cv2.fillPoly(observed, [np.array(polygon, dtype=np.int32)], 127)
        observed = cv2.morphologyEx(
            observed, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
        )

    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    obstacle_mask = points[:, 2] > 0.30
    occupied[rows[in_bounds & obstacle_mask], cols[in_bounds & obstacle_mask]] = 255
    occupied = cv2.dilate(occupied, np.ones((3, 3), dtype=np.uint8), iterations=1)

    observed[occupied > 0] = 0
    occupancy = observed.copy()
    occupancy[occupied > 0] = 255

    return occupancy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a camera BEV mosaic and LiDAR occupancy grid from nuScenes."
    )
    parser.add_argument("--dataset-root", default="v1.0-mini", help="Path to the nuScenes mini root.")
    parser.add_argument("--sample-index", type=int, default=0, help="Keyframe sample index to visualize.")
    parser.add_argument("--sample-token", default=None, help="Optional sample token override.")
    parser.add_argument("--x-min", type=float, default=-18.0, help="Rear extent in meters.")
    parser.add_argument("--x-max", type=float, default=30.0, help="Front extent in meters.")
    parser.add_argument("--y-min", type=float, default=-20.0, help="Right extent in meters.")
    parser.add_argument("--y-max", type=float, default=20.0, help="Left extent in meters.")
    parser.add_argument("--resolution", type=float, default=0.08, help="Meters per pixel.")
    parser.add_argument("--bev-output", default="final_bev.jpg", help="Output path for the stitched BEV image.")
    parser.add_argument(
        "--occupancy-output",
        default="occupancy_grid.png",
        help="Output path for the grayscale occupancy grid.",
    )
    parser.add_argument("--show", action="store_true", help="Display the generated images in OpenCV windows.")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    meta_root = dataset_root / "v1.0-mini"
    if not meta_root.exists():
        raise FileNotFoundError(f"Could not find nuScenes metadata under: {meta_root}")

    indexes = build_indexes(meta_root)
    sample_token, bundle = build_sample_bundle(
        indexes, sample_index=args.sample_index, sample_token=args.sample_token
    )

    x_limits = (args.x_min, args.x_max)
    y_limits = (args.y_min, args.y_max)

    bev, confidence, vehicle_mask = build_camera_bev(
        dataset_root=dataset_root,
        bundle=bundle,
        x_limits=x_limits,
        y_limits=y_limits,
        resolution=args.resolution,
    )

    occupancy = build_occupancy_grid(
        dataset_root=dataset_root,
        bundle=bundle,
        x_limits=x_limits,
        y_limits=y_limits,
        resolution=args.resolution,
    )

    coverage_mask = (confidence >= 0.18) | (occupancy > 0)
    bev[~coverage_mask] = 0
    bev = cv2.GaussianBlur(bev, (3, 3), 0)
    bev[vehicle_mask] = (40, 40, 40)
    occupancy[vehicle_mask] = 80

    cv2.imwrite(args.bev_output, bev)
    cv2.imwrite(args.occupancy_output, occupancy)

    print(f"Sample token: {sample_token}")
    print(f"Saved BEV image to: {Path(args.bev_output).resolve()}")
    print(f"Saved occupancy grid to: {Path(args.occupancy_output).resolve()}")

    if args.show:
        cv2.imshow("Camera BEV", bev)
        cv2.imshow("Occupancy Grid", occupancy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        raise ImportError(
            "final_bev.py requires opencv-python in the interpreter you use to run it."
        ) from exc
