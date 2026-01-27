"""
body_axis_flow_core.py

Purpose
-------
This script computes dense optical flow between consecutive video frames and
quantifies motion along predefined body axes within a fixed region of interest (ROI).

Specifically, for each frame:
  1) Dense optical flow is computed using the Farnebäck method.
  2) The 2D flow vectors (fx, fy) are projected onto body-axis unit vectors
     ex and ey, which are estimated upstream.
  3) The projected flow components are spatially averaged within an ROI polygon.
  4) Frame-wise features are saved to a CSV file.

Inputs
------
- video_path : str
    Path to the input video file.
- inter_npz : str
    Path to an upstream NPZ file containing:
        - time_all : array (T,)
            Time stamps corresponding to body-axis estimates.
        - fps : float
            Frame rate used in upstream processing.
        - ex, ey : arrays (T, 2)
            Unit vectors defining the body coordinate system.
- roi_polygon_xy : ndarray (N, 2)
    Polygon defining the ROI in image coordinates.

Outputs
-------
- out_csv : str
    CSV file with per-frame motion features:
        frame, t_sec, skel_idx, axes_ok, vx_body, vy_body, mag_body
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cv2


# ==========================================================
# Farnebäck optical flow parameters
#
# These parameters control the spatial and temporal sensitivity
# of dense optical flow estimation.
# In practice, winsize is the parameter most often adjusted.
# ==========================================================
FB_PARAMS = dict(
    pyr_scale=0.5,     # Image pyramid scaling factor
    levels=3,          # Number of pyramid levels
    winsize=15,        # Averaging window size (pixels)
    iterations=3,      # Iterations per pyramid level
    poly_n=5,          # Size of the pixel neighborhood
    poly_sigma=1.2,    # Standard deviation of the Gaussian
    flags=0
)


# ==========================================================
# Utility functions
# ==========================================================
def open_video(video_path: str, fallback_fps: float) -> tuple[cv2.VideoCapture, float, int, int]:
    """
    Open a video file and extract basic metadata.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    fallback_fps : float
        Frame rate used if the video file does not provide a valid FPS.

    Returns
    -------
    cap : cv2.VideoCapture
        OpenCV video capture object.
    fps : float
        Frames per second.
    W, H : int
        Frame width and height.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"VideoCapture failed: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = float(fallback_fps)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, float(fps), W, H


def build_roi_mask(H: int, W: int, roi_polygon_xy: np.ndarray) -> np.ndarray:
    """
    Convert an ROI polygon into a boolean mask.

    Parameters
    ----------
    H, W : int
        Image height and width.
    roi_polygon_xy : ndarray (N, 2)
        Polygon vertices in image coordinates.

    Returns
    -------
    mask : ndarray (H, W), bool
        True for pixels inside the ROI.
    """
    poly = np.asarray(roi_polygon_xy, dtype=np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    return mask.astype(bool)


def frame_time_sec(cap: cv2.VideoCapture, frame_idx: int, fps: float) -> float:
    """
    Obtain the time stamp of the current frame in seconds.

    Priority is given to CAP_PROP_POS_MSEC.
    If unavailable, time is estimated from frame index and FPS.
    """
    t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if t_msec is not None and t_msec > 0:
        return float(t_msec) / 1000.0
    return float(frame_idx) / float(fps)


def skel_index_from_time(t_sec: float, time_all: np.ndarray) -> int:
    """
    Map a video time stamp to the corresponding upstream index.

    The selected index satisfies:
        time_all[idx] <= t_sec
    and is the largest such index (no look-ahead).

    This ensures temporal causality between video frames
    and body-axis estimates.
    """
    idx = int(np.searchsorted(time_all, t_sec, side="right") - 1)
    idx = int(np.clip(idx, 0, len(time_all) - 1))
    return idx


def compute_roi_mean_body_flow(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    roi_mask: np.ndarray,
    fb_params: dict
) -> tuple[float, float, float]:
    """
    Compute ROI-averaged optical flow features in the body coordinate system.

    Steps
    -----
    1) Compute dense optical flow between consecutive grayscale frames.
    2) Project 2D flow vectors onto body-axis unit vectors ex and ey.
    3) Compute spatial means of the projected flow components within the ROI.

    Parameters
    ----------
    prev_gray, gray : ndarray (H, W)
        Consecutive grayscale frames.
    ex, ey : ndarray (2,)
        Unit vectors defining the body coordinate system.
    roi_mask : ndarray (H, W), bool
        ROI mask.
    fb_params : dict
        Farnebäck optical flow parameters.

    Returns
    -------
    vx_mean : float
        Mean flow component along ex within the ROI.
    vy_mean : float
        Mean flow component along ey within the ROI.
    mag_mean : float
        Mean flow magnitude within the ROI.
    """
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **fb_params)

    # Image-coordinate flow components
    fx = flow[..., 0]
    fy = flow[..., 1]

    # Projection onto body axes
    fx_body = fx * float(ex[0]) + fy * float(ex[1])
    fy_body = fx * float(ey[0]) + fy * float(ey[1])

    mag_body = cv2.magnitude(fx_body, fy_body)

    vx_mean = float(np.nanmean(fx_body[roi_mask]))
    vy_mean = float(np.nanmean(fy_body[roi_mask]))
    mag_mean = float(np.nanmean(mag_body[roi_mask]))

    return vx_mean, vy_mean, mag_mean


# ==========================================================
# Main processing function
# ==========================================================
def run_body_axis_flow_core(
    video_path: str,
    inter_npz: str,
    roi_polygon_xy: np.ndarray,
    out_csv: str,
) -> None:
    """
    Run body-axis optical flow analysis and save results to CSV.
    """
    # --- Load upstream data ---
    dat = np.load(inter_npz, allow_pickle=True)

    time_all = np.asarray(dat["time_all"], dtype=float)
    fps_npz = float(dat["fps"])
    ex_all = np.asarray(dat["ex"], dtype=float)
    ey_all = np.asarray(dat["ey"], dtype=float)

    # --- Open video ---
    cap, fps_vid, W, H = open_video(video_path, fallback_fps=fps_npz)

    # --- Build ROI mask ---
    roi_mask = build_roi_mask(H, W, roi_polygon_xy)

    rows = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t_sec = frame_time_sec(cap, frame_idx, fps_vid)
        skel_idx = skel_index_from_time(t_sec, time_all)

        ex = ex_all[skel_idx]
        ey = ey_all[skel_idx]
        axes_ok = bool(np.isfinite(ex).all() and np.isfinite(ey).all())

        vx = np.nan
        vy = np.nan
        mag = np.nan

        # Optical flow is computed only when body axes are valid
        # and a previous frame is available.
        if axes_ok and (prev_gray is not None):
            vx, vy, mag = compute_roi_mean_body_flow(
                prev_gray, gray, ex, ey, roi_mask, FB_PARAMS
            )

        rows.append([frame_idx, t_sec, skel_idx, int(axes_ok), vx, vy, mag])

        prev_gray = gray
        frame_idx += 1

    cap.release()

    # --- Save CSV ---
    df = pd.DataFrame(
        rows,
        columns=["frame", "t_sec", "skel_idx", "axes_ok", "vx_body", "vy_body", "mag_body"]
    )
    df.to_csv(out_csv, index=False)


# ==========================================================
# Example usage
# ==========================================================
if __name__ == "__main__":
    video_path = "input.mp4"
    inter_npz = "skeleton_pc1.npz"

    roi_polygon_xy = np.array(
        [
            [100, 100],
            [500, 120],
            [520, 380],
            [120, 400],
        ],
        dtype=float
    )

    out_csv = "flow.csv"

    run_body_axis_flow_core(
        video_path=video_path,
        inter_npz=inter_npz,
        roi_polygon_xy=roi_polygon_xy,
        out_csv=out_csv,
    )

    print("Saved:", out_csv)
