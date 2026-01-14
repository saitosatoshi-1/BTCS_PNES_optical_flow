# =============================================================
# Body-axis Optical Flow in a Fixed ROI (manual polygon)
#
# For each video frame, this cell:
#   1) computes dense optical flow (Farnebäck) between consecutive frames
#   2) loads precomputed body axes (ex, ey) from an upstream NPZ (no re-loading of track.npz)
#   3) projects 2D flow vectors onto the body axes (ex, ey)
#   4) averages the projected flow within a fixed ROI mask
#   5) saves:
#        - an overlay MP4 for QC (flow visualization inside ROI)
#        - a CSV for analysis (vx_body, vy_body, mag_body)
#        - a QC summary CSV (counts of invalid frames, meta info)
#
# Important notes (practical pitfalls)
#   - Optical flow requires a previous frame, so the first frame is NaN by design.
#   - If ex/ey are NaN in a frame, upstream body-axis estimation failed,
#     so we output NaN for that frame.
#   - We do NOT re-load track.npz here, to keep consistency and improve speed.
# =============================================================

import cv2
import numpy as np
import pandas as pd
import os


# =============================================================
# Input / output paths
# =============================================================
video_path = "/content/FBTCS_qt.mp4"

# Upstream intermediate file containing time_all, fps, ex, ey, etc.
inter_npz  = "/content/skeleton_pc1.npz"

out_video  = "/content/flow.mp4"              # QC video (flow overlay)
out_csv    = "/content/flow.csv"              # per-frame features
qc_csv     = "/content/flow_qc_summary.csv"   # QC summary


# =============================================================
# Load upstream signals (body axes and time base)
# =============================================================
dat2 = np.load(inter_npz, allow_pickle=True)

time_all = dat2["time_all"]          # (T,) seconds
fps_npz  = float(dat2["fps"])        # upstream FPS (trusted reference)
ex_all   = dat2["ex"]                # (T,2) unit vector, may contain NaN
ey_all   = dat2["ey"]                # (T,2) unit vector, may contain NaN

mask_t   = dat2["mask_t"]            # (T,) optional analysis mask
meta     = dat2["meta"][0]           # dict (stored as object array)
print("[meta]", meta)


# =============================================================
# Farnebäck optical flow parameters
# =============================================================
FB_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)


# =============================================================
# ROI polygon (bed area)
# - poly_rot_global is expected to be defined upstream,
#   e.g., from an interactive polygon editor cell.
# =============================================================
assert poly_rot_global is not None, "BED polygon not set"
ROI_POINTS = poly_rot_global.astype(np.float32)


# =============================================================
# Helper functions
# =============================================================
def open_video_io(video_path, out_video, fallback_fps):
    """
    Open an input video and create an output VideoWriter.

    fallback_fps is used when the input video does not provide FPS reliably.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"VideoCapture failed: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = fallback_fps

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        out_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )
    return cap, out, fps, W, H


def build_roi_mask(H, W, roi_points):
    """
    Create a boolean mask that is True inside the ROI polygon.
    """
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [roi_points.astype(np.int32)], 1)
    return mask.astype(bool)


def skeleton_index_from_t(t_sec, time_all):
    """
    Map video time (seconds) to the nearest *past* skeleton index.

    We choose the maximum index where time_all[idx] <= t_sec (no look-ahead).
    Then clip into [0, len(time_all)-1].
    """
    ai = int(np.searchsorted(time_all, t_sec, side="right") - 1)
    ai = int(np.clip(ai, 0, len(time_all) - 1))
    return ai


def get_frame_time_sec(cap, frame_idx, fps):
    """
    Get the timestamp of the current frame in seconds.

    If CAP_PROP_POS_MSEC is available, use it.
    Otherwise, fall back to frame_idx / fps.
    """
    t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if t_msec and t_msec > 0:
        return t_msec / 1000.0
    return frame_idx / fps


def compute_flow_features(flow_prev, gray, ex, ey, mask_bool, fb_params):
    """
    Compute optical flow and ROI-mean features in the body coordinate system.

    Parameters
    ----------
    flow_prev : ndarray (H,W), previous grayscale frame
    gray      : ndarray (H,W), current grayscale frame
    ex, ey    : ndarray (2,), body-axis unit vectors
    mask_bool : ndarray (H,W), True inside ROI
    fb_params : dict, Farnebäck parameters

    Returns
    -------
    vx_mean, vy_mean, mag_mean : float
        ROI-mean projected flow along body axes and magnitude.
    flow_color : ndarray (H,W,3), BGR
        Color visualization of projected flow (HSV -> BGR).
    """
    # Dense optical flow (pixel-wise 2D motion vectors)
    flow = cv2.calcOpticalFlowFarneback(flow_prev, gray, None, **fb_params)

    # flow[...,0] is x-component, flow[...,1] is y-component in image coordinates
    fx = flow[..., 0]
    fy = flow[..., 1]

    # Project image-coordinate flow (fx, fy) onto body axes (ex, ey)
    # Dot product: (fx, fy) · ex  and  (fx, fy) · ey
    fx_body = fx * ex[0] + fy * ex[1]
    fy_body = fx * ey[0] + fy * ey[1]

    mag_body = cv2.magnitude(fx_body, fy_body)

    # ROI mean (flattened by mask_bool)
    vx_mean  = float(np.nanmean(fx_body[mask_bool]))
    vy_mean  = float(np.nanmean(fy_body[mask_bool]))
    mag_mean = float(np.nanmean(mag_body[mask_bool]))

    # Visualization:
    #   Hue   = direction (angle)
    #   Value = magnitude (brightness)
    #
    # Use arctan2(y, x) to handle all quadrants and x=0 safely.
    ang = np.arctan2(fy_body, fx_body)

    hsv = np.zeros((*gray.shape, 3), np.uint8)
    hsv[..., 1] = 255  # saturation

    # OpenCV HSV hue range is [0, 180], not [0, 360].
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)

    # Normalize magnitude into [0, 255] for brightness
    hsv[..., 2] = cv2.normalize(mag_body, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return vx_mean, vy_mean, mag_mean, flow_color


def overlay_flow(frame, flow_color, mask_bool, roi_points):
    """
    Overlay the flow visualization only inside the ROI.
    Also draw the ROI polygon for QC.
    """
    overlay = frame.copy()

    # Blend original image and flow visualization
    blended = cv2.addWeighted(frame, 0.5, flow_color, 0.5, 0)

    # Replace only ROI pixels
    overlay[mask_bool] = blended[mask_bool]

    # Draw ROI polygon boundary
    cv2.polylines(overlay, [roi_points.astype(int)], True, (0, 255, 255), 2)

    return overlay


def darken_frame(frame, alpha=0.4):
    """
    Darken the whole frame (useful when features are unavailable).

    alpha : 0..1
      0.0 -> no change
      0.6 -> noticeably dark (often good for QC)
      0.8 -> very dark
    """
    black = np.zeros_like(frame)
    return cv2.addWeighted(frame, 1 - alpha, black, alpha, 0)


# =============================================================
# Open video and prepare ROI mask
# =============================================================
cap, out, fps_vid, W, H = open_video_io(video_path, out_video, fps_npz)
mask_bool = build_roi_mask(H, W, ROI_POINTS)


# =============================================================
# Main loop
# =============================================================
rows = []
flow_prev = None
frame_idx = 0

# QC counters
cnt_flow_none = 0       # first frame (no previous frame)
cnt_axes_invalid = 0    # frames where ex/ey are invalid (NaN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Current frame time (seconds)
    t_sec = get_frame_time_sec(cap, frame_idx, fps_vid)

    # Synchronize with upstream time axis
    ai = skeleton_index_from_t(t_sec, time_all)

    # Use upstream body axes
    ex = ex_all[ai]
    ey = ey_all[ai]
    axes_ok = bool(np.isfinite(ex).all() and np.isfinite(ey).all())

    # Default outputs (NaN + darkened QC frame)
    vx_mean = np.nan
    vy_mean = np.nan
    mag_mean = np.nan
    overlay = darken_frame(frame, alpha=0.6)

    # Case 1: body axes invalid -> output NaN
    if not axes_ok:
        cnt_axes_invalid += 1
        cv2.polylines(overlay, [ROI_POINTS.astype(int)], True, (0, 255, 255), 2)

    # Case 2: first frame -> no optical flow available
    elif flow_prev is None:
        cnt_flow_none += 1
        cv2.polylines(overlay, [ROI_POINTS.astype(int)], True, (0, 255, 255), 2)

    # Case 3: normal computation
    else:
        vx_mean, vy_mean, mag_mean, flow_color = compute_flow_features(
            flow_prev, gray, ex, ey, mask_bool, FB_PARAMS
        )
        overlay = overlay_flow(frame, flow_color, mask_bool, ROI_POINTS)

    out.write(overlay)

    # Save per-frame results for later analysis
    rows.append([frame_idx, t_sec, ai, int(axes_ok), vx_mean, vy_mean, mag_mean])

    # Update previous frame for optical flow
    flow_prev = gray
    frame_idx += 1

cap.release()
out.release()


# =============================================================
# Save per-frame features to CSV
# =============================================================
df_out = pd.DataFrame(
    rows,
    columns=["frame", "t_sec", "skel_idx", "axes_ok", "vx_body", "vy_body", "mag_body"]
)
df_out.to_csv(out_csv, index=False)
print("[OK] flow CSV saved:", out_csv)


# =============================================================
# Save QC summary
# =============================================================
qc = {
    "video_path": video_path,
    "intermediate_npz": inter_npz,
    "fps_video": float(fps_vid),
    "fps_npz": float(fps_npz),
    "n_frames_out": int(frame_idx),
    "cnt_flow_none_firstframe": int(cnt_flow_none),
    "cnt_axes_invalid": int(cnt_axes_invalid),
    "meta": meta,
}
pd.DataFrame([qc]).to_csv(qc_csv, index=False)

print("[OK] QC summary saved:", qc_csv)
print("[QC] cnt_axes_invalid:", cnt_axes_invalid, ", cnt_flow_none:", cnt_flow_none)
