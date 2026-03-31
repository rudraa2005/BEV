# Lift-Splat-Shoot BEV (Bird's-Eye-View)

This repository contains an end-to-end implementation of the Lift-Splat-Shoot architecture for translating multi-view camera images into a comprehensive Bird's-Eye-View (BEV) occupancy grid. It transitions away from classic homography mapping to a robust, deep-learning based 3D frustum projection.

## Core Features
- **Frustum Lifting:** Accurately projects 2D image pixels into a 3D frustum voxel space using dynamic depth distribution bins.
- **Robust Generalization:** Employs Color Jittering, heavy Rotation/Resize augmentations, and Decoder regularizations (`Dropout2d`) to actively prevent dataset memorization and overfitting.
- **Voxel Pooling:** Utilizes `QuickCumsum` techniques to collapse 3D representations onto the 2D BEV plane efficiently.
- **TensorBoard Tracking:** Native integration for tracking validation metrics and automatically snapping the peak `model_best.pt` architecture.

## Getting Started

### 1. Data Preparation
You must download and extract the NuScenes dataset (`v1.0-mini`) pointing the configurations to your local map. Ensure your environment has the corresponding `nuscenes-devkit` installed via `requirements.txt`.

### 2. Training the Model
To begin the training pipeline natively:
```bash
python train.py --nepochs 80 --bsz 2
```

### 3. Evaluating & Visualization
Once the peak checkpoint is saved, you can cleanly generate the 2D Occupancy patches from 6-camera input arrays using the evaluation script with Morphological Noise thresholds.
```bash
python evaluate.py --modelf runs/model_best.pt --threshold 0.6
```

## Monitoring Performance
You can actively trace and monitor your losses and IoU tracking by pointing TensorBoard to the running checkpoint logs:
```bash
tensorboard --logdir=./runs
```
