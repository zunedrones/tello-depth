# tello-depth

Real-time monocular depth estimation for the DJI Tello drone, using [LibreYOLO](https://github.com/LibreYOLO/libreyolo) depth models. The drone is piloted from the keyboard while the camera feed is displayed side-by-side with the depth map, and the center-region distance is measured.

## Features

- **Keyboard piloting** of the Tello (takeoff, land, all axes, yaw rotation, emergency stop)
- **Real-time depth estimation** with LibreYOLO models, running in a separate thread so video and controls stay responsive
- **Center-region depth measurement**: a square at the center of the frame; the average depth of the area is displayed (higher relative value = closer)
- **Metric distance** via a saved scale factor, calibrated with a one-point reference (key `C`) or automatically with the drone's TOF + IMU during a short forward maneuver (key `V`)
- **Camera calibration** (`config/tello_02.yaml`): intrinsics are loaded and lens distortion is corrected before depth inference
- **Available depth models** (download automatically on first use):
  - `LibreMiDaSs-depth.pt` — MiDaS Small (EfficientNet-Lite3 encoder, 256 px input). Fastest, lightest quality. Uses extra dependency `timm` (`pip install 'libreyolo[midas]'`)
  - `LibreDepthAnythingV2s-depth.pt` — Depth Anything V2 Small (ViT-S + DPT head, 518 px input). Better quality, moderate CPU cost (default)
  - `LibreZipDepthb-depth.pt` — ZipDepth Base (lightweight CNN distilled from Depth Anything V2 Large, 384 px input). Fast, good quality
  - `LibreMiDaSl-depth.pt` — MiDaS Large (DPT-Large, 384 px input). Much better, heavy on CPU (~1.3 GB weights)
  - Depth Anything V2 Base/Large/Giant and Depth Anything 3: **not redistributed** by LibreYOLO (CC-BY-NC license on Base/Large/Giant); they require manual conversion of the upstream checkpoints

  All models output **relative inverse depth** (no metric unit): higher values mean closer. Select the model in `main.py` via `DepthEstimator("<checkpoint>")`.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build (skip if GPU)
pip install "libreyolo[midas]" libtello-zune opencv-python numpy
```

## Usage

1. Connect to the Tello Wi-Fi network.
2. Run:

```bash
python main.py
```

## Keyboard controls

| Key | Action |
|-----|--------|
| `T` | Takeoff |
| `L` | Land |
| `W` / `S` | Forward / Backward |
| `A` / `D` | Left / Right |
| `R` / `F` | Up / Down |
| `Q` / `E` | Rotate counter-clockwise / clockwise (yaw) |
| `Space` | Emergency stop (motors cut immediately!) |
| `C` | Instant scale calibration: point the center square at an object at `KNOWN_DISTANCE_M` meters (see `main.py`) and press |
| `V` | Automatic calibration: drone verifies TOF height, flies ~50 cm forward while IMU/odometry measures the displacement, then computes the metric scale |
| `ESC` | Exit |

The metric scale is saved to `config/depth_scale.json` and reloaded on the next run, so calibration is only needed once per depth model.

## Project structure

```
main.py                  # Main loop: video capture, overlays, keyboard control
keyboard_controller.py   # Keyboard mapping and RC commands to the drone
depth_estimator.py       # Depth inference thread (LibreYOLO) + center-region averaging
camera_calibration.py    # Camera intrinsics from config/tello_02.yaml + metric scale storage
motion_calibration.py    # Automatic scale calibration via TOF + IMU forward maneuver
config/tello_02.yaml     # Camera calibration (intrinsics + distortion, 960x720)
```

## Notes

- The Tello's TOF sensor points **downward** (measures height above the ground); it cannot measure what the forward camera sees. It is used only as a safety check before the automatic calibration maneuver, and its reading is displayed on screen as `Height`.
- Monocular depth models have an inherent scale ambiguity; metric conversion requires a scale factor, which the `C` and `V` calibrations provide.
- Inference runs on CPU. Resolution can be changed with `tello.set_image_size((width, height))` in `main.py`; lower resolutions improve display/keyboard FPS (the model letterboxes internally anyway, so inference time barely changes).
