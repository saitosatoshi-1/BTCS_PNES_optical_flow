# ==========================================================
# Person1 Tracking + Pose Estimation
#
# What this cell does.
#  1. Read a video frame-by-frame.
#  2. Run YOLO Pose to detect persons and 17 keypoints (COCO format).
#  3. Keep only detections whose "person center" falls inside a bed polygon gate.
#  4. If exactly one candidate is inside the gate, lock it as "person1".
#  5. Save keypoints over time to an NPZ file.
#  6. Draw the skeleton on an output video.
#  7. If detection fails, do not crash. Fill with NaN or carry forward the last valid result.
#
# Typical use case.
#  - Create stable tracking of the bed patient (person1) in VEEG videos.
#  - Use the saved keypoints for downstream motion analysis.
#
# Privacy note.
#  - This cell optionally applies face mosaic using facial keypoints (nose, eyes, ears).
# ==========================================================


# ----------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------
import os
import numpy as np
import cv2
from ultralytics import YOLO


# ----------------------------------------------------------
# 2. Paths and constants
# ----------------------------------------------------------
video_path   = "/content/FBTCS_qt.mp4"
out_try_path = "/content/track.mp4"
out_qt_path  = "/content/track_qt.mp4"
npz_out      = "/content/track.npz"

# Minimum confidence threshold for YOLO keypoints.
# Keypoints below this will be treated as missing (NaN).
CONF_MIN = 0.3

# Face mosaic settings (tune if needed).
FACE_MOSAIC_BLOCK = 14   # larger = coarser mosaic
FACE_CONF_THR = 0.3      # confidence threshold for facial keypoints

# COCO 17-keypoint skeleton connections (which joints to connect by lines).
SKELETON = [
    (0, 1), (1, 3), (0, 2), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


# ----------------------------------------------------------
# 3. Utility functions (small building blocks)
#    These are called by other functions, so we define them first.
# ----------------------------------------------------------
def center_from_kpts(kp_xy):
    """
    Compute a representative person center from keypoints.

    Parameters
    ----------
    kp_xy : ndarray, shape (17, 2)
        Keypoint coordinates (x, y).
        Low-confidence keypoints are expected to be NaN already.

    Returns
    -------
    center : ndarray, shape (2,) or None
        Mean (x, y) of all valid (non-NaN) keypoints.
        Returns None if no valid keypoints are available.
    """
    valid = ~np.isnan(kp_xy).any(axis=1)
    if not np.any(valid):
        return None
    return np.nanmean(kp_xy[valid], axis=0)


def inside_gate(x, y):
    """
    Check whether a point (x, y) lies inside the bed polygon gate.

    This uses OpenCV pointPolygonTest.
    gate_poly must be a global polygon with shape (N, 1, 2).
    """
    return cv2.pointPolygonTest(
        gate_poly,
        (float(x), float(y)),
        False
    ) >= 0


def compute_head_point(kp_xy, kp_c, conf_thr=0.3):
    """
    Determine a single representative head point.

    Strategy
    ----------
    1. If nose (index 0) is valid, use it.
    2. Otherwise, use the mean of valid facial points: eyes and ears (indices 1-4).
    3. If nothing is valid, return NaN.

    Returns
    -------
    head_xy : ndarray, shape (2,)
        Head point (x, y) or [NaN, NaN].
    head_src : str
        "nose", "face", or "none".
    """
    if kp_c[0] >= conf_thr and np.isfinite(kp_xy[0]).all():
        return kp_xy[0].copy(), "nose"

    face_idx = [1, 2, 3, 4]  # eyes, ears
    valid = [
        i for i in face_idx
        if kp_c[i] >= conf_thr and np.isfinite(kp_xy[i]).all()
    ]
    if len(valid) > 0:
        return np.mean(kp_xy[valid], axis=0), "face"

    return np.array([np.nan, np.nan]), "none"


def draw_skeleton(img, kp_xy, color=(0, 0, 255), th=2):
    """
    Draw the skeleton on a frame for visual QC.

    img is an OpenCV image (BGR).
    kp_xy is (17,2) with NaN for missing points.
    """
    for a, b in SKELETON:
        xa, ya = kp_xy[a]
        xb, yb = kp_xy[b]
        if (np.isfinite(xa) and np.isfinite(ya)
            and np.isfinite(xb) and np.isfinite(yb)):
            cv2.line(
                img,
                (int(xa), int(ya)),
                (int(xb), int(yb)),
                color,
                th
            )


def clip_box(x1, y1, x2, y2, W, H):
    """
    Clip a bounding box so it stays inside the image.
    """
    x1 = int(max(0, min(W - 2, x1)))
    y1 = int(max(0, min(H - 2, y1)))
    x2 = int(max(x1 + 1, min(W, x2)))
    y2 = int(max(y1 + 1, min(H, y2)))
    return x1, y1, x2, y2


def apply_mosaic(img, x1, y1, x2, y2, block=14):
    """
    Apply pixelation mosaic to a rectangle region.
    Larger block means coarser mosaic.
    """
    roi = img[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if h <= 2 or w <= 2:
        return img

    small_w = max(1, w // block)
    small_h = max(1, h // block)

    roi_small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    roi_mos   = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)

    out = img.copy()
    out[y1:y2, x1:x2] = roi_mos
    return out


def face_bbox_from_kpts(kp_xy, kp_c, W, H, conf_thr=0.3, margin_x=0.6, margin_y=0.9):
    """
    Estimate a face bounding box from facial keypoints (nose, eyes, ears).

    Returns
    -------
    bbox : tuple (x1, y1, x2, y2) or None
        Returns None if too few facial points are available.
    """
    face_idx = [0, 1, 2, 3, 4]  # nose, eyes, ears
    pts = []
    for i in face_idx:
        x, y = kp_xy[i]
        c = kp_c[i]
        if (c >= conf_thr) and np.isfinite(x) and np.isfinite(y):
            pts.append([x, y])

    # If points are insufficient, the bbox becomes unstable.
    if len(pts) < 2:
        return None

    pts = np.asarray(pts, float)
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    bw = max(2.0, x_max - x_min)
    bh = max(2.0, y_max - y_min)

    x1 = x_min - margin_x * bw
    x2 = x_max + margin_x * bw
    y1 = y_min - margin_y * bh
    y2 = y_max + margin_y * bh

    return clip_box(x1, y1, x2, y2, W, H)


# ----------------------------------------------------------
# 4. Per-frame functions (medium-level logic built from utilities)
# ----------------------------------------------------------
def yolo_pose(frame):
    """
    Run YOLO Pose on a single frame.

    Returns
    -------
    boxes : ndarray, shape (N,4)
        Bounding boxes for detected persons, each row is (x1, y1, x2, y2).
    kpts_all : ndarray, shape (N,17,3)
        Keypoints for each person, each keypoint is (x, y, confidence).
    """
    res = pose_model.predict(
        frame,
        verbose=False,
        conf=CONF_MIN
    )[0]

    boxes = (
        res.boxes.xyxy.cpu().numpy()
        if res.boxes is not None else np.zeros((0, 4))
    )

    kpts_all = (
        res.keypoints.data.cpu().numpy()
        if res.keypoints is not None else np.zeros((0, 17, 3))
    )

    return boxes, kpts_all


def make_candidates(boxes, kpts_all):
    """
    Collect persons whose representative center lies inside the bed polygon gate.

    We first compute a "person center" from keypoints (mean of valid points).
    If keypoints are all missing, fall back to bbox center.

    Returns
    -------
    candidates : list of dict
        Each dict contains:
        - i      : person index in YOLO output
        - center : ndarray (2,)
        - kp_xy  : ndarray (17,2) with NaN for low-confidence points
        - kp_c   : ndarray (17,)
    """
    candidates = []

    for i in range(kpts_all.shape[0]):
        kp_xy = kpts_all[i, :, :2].copy()
        kp_c  = kpts_all[i, :, 2].copy()

        # Treat low-confidence points as missing.
        kp_xy[kp_c < CONF_MIN] = np.nan

        center = center_from_kpts(kp_xy)

        # If skeleton center is undefined, use bbox center.
        if center is None:
            if boxes.shape[0] > i:
                x1, y1, x2, y2 = boxes[i]
                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            else:
                continue

        if inside_gate(center[0], center[1]):
            candidates.append(
                dict(i=i, center=center, kp_xy=kp_xy, kp_c=kp_c)
            )

    return candidates


def pack_result(chosen):
    """
    Convert the chosen person dict into arrays suitable for saving.

    Returns
    -------
    row_k : ndarray, shape (17,3)
        [x, y, confidence] for each keypoint.
    head_xy : ndarray, shape (2,)
        Representative head point (x, y).
    head_src : str
        "nose", "face", or "none".
    """
    row_k = np.stack(
        [chosen["kp_xy"][:, 0], chosen["kp_xy"][:, 1], chosen["kp_c"]],
        axis=1
    )

    head_xy, head_src = compute_head_point(
        chosen["kp_xy"], chosen["kp_c"], CONF_MIN
    )

    return row_k, head_xy, head_src


def missing_result(last_valid_row_k, last_valid_head_xy, last_valid_head_src, carry=True):
    """
    Fill a frame where person1 cannot be uniquely determined.

    carry=True:
      Use the last locked result if available. This is useful for continuity.
    carry=False:
      Always return NaN.

    Returns
    -------
    row_k : (17,3)
    head_xy : (2,)
    head_src : str
    state_str : str, one of "CARRY" or "MISSING"
    """
    if carry and (last_valid_row_k is not None):
        row_k = last_valid_row_k.copy()

        if last_valid_head_xy is not None:
            head_xy = last_valid_head_xy.copy()
            head_src = last_valid_head_src if last_valid_head_src is not None else "carried"
        else:
            head_xy = np.array([np.nan, np.nan], dtype=float)
            head_src = "carried_nohead"

        return row_k, head_xy, head_src, "CARRY"

    row_k = np.full((17, 3), np.nan, dtype=float)
    head_xy = np.array([np.nan, np.nan], dtype=float)
    return row_k, head_xy, "missing", "MISSING"


# ----------------------------------------------------------
# 5. Initialization (video, model, polygon gate)
# ----------------------------------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"VideoCapture failed: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 30.0

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    out_try_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H)
)

# YOLO Pose model (choose size based on your compute budget).
pose_model = YOLO("yolo11x-pose.pt")

# Bed polygon gate.
# This code expects poly_rot_global to be defined upstream,
# for example from an interactive polygon editor cell.
assert poly_rot_global is not None, "BED polygon not set. Please run the polygon editor cell first."

priority_coords = poly_rot_global.astype(np.float32)
gate_poly = priority_coords.reshape(-1, 1, 2)

polygon_cx = float(priority_coords[:, 0].mean())
polygon_cy = float(priority_coords[:, 1].mean())


# ----------------------------------------------------------
# 6. Main loop (frame-by-frame processing)
# ----------------------------------------------------------
all_kpts = []         # per-frame skeleton: (17,3)
all_state = []        # "LOCKED", "CARRY", "MISSING", "AMBIGUOUS"
all_idx = []          # person index from YOLO, -1 if not determined

all_head_xy = []      # per-frame head coordinate (2,)
all_head_source = []  # "nose", "face", "none", etc.

last_valid_row_k = None
last_valid_head_xy = None
last_valid_head_src = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, kpts_all = yolo_pose(frame)

    # Case A: no persons detected.
    if kpts_all.shape[0] == 0:
        row_k, head_xy, head_src, st = missing_result(
            last_valid_row_k,
            last_valid_head_xy,
            last_valid_head_src,
            carry=True
        )
        all_state.append(st)
        all_idx.append(-1)

    else:
        # Filter persons inside the bed gate.
        candidates = make_candidates(boxes, kpts_all)

        # Case B: nobody is inside the gate.
        if len(candidates) == 0:
            row_k, head_xy, head_src, st = missing_result(
                last_valid_row_k,
                last_valid_head_xy,
                last_valid_head_src,
                carry=True
            )
            all_state.append(st)
            all_idx.append(-1)

        else:
            # Case C: ambiguous, more than one candidate inside the gate.
            if len(candidates) != 1:
                row_k, head_xy, head_src, st = missing_result(
                    last_valid_row_k,
                    last_valid_head_xy,
                    last_valid_head_src,
                    carry=True
                )
                all_state.append("AMBIGUOUS")
                all_idx.append(-1)

            # Case D: exactly one candidate, lock as person1.
            else:
                chosen = candidates[0]
                row_k, head_xy, head_src = pack_result(chosen)

                all_state.append("LOCKED")
                all_idx.append(chosen["i"])

                # Face mosaic first (privacy), then draw skeleton for QC.
                bbox = face_bbox_from_kpts(
                    chosen["kp_xy"], chosen["kp_c"],
                    W, H,
                    conf_thr=FACE_CONF_THR,
                    margin_x=0.6,
                    margin_y=0.9
                )
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    frame = apply_mosaic(frame, x1, y1, x2, y2, block=FACE_MOSAIC_BLOCK)

                draw_skeleton(frame, chosen["kp_xy"])

                # Update last valid result for carry-forward.
                last_valid_row_k = row_k
                last_valid_head_xy = head_xy
                last_valid_head_src = head_src

    # Save outputs for this frame.
    all_kpts.append(row_k)
    all_head_xy.append(head_xy)
    all_head_source.append(head_src)

    out.write(frame)

cap.release()
out.release()


# ----------------------------------------------------------
# 7. Save results to NPZ
# ----------------------------------------------------------
kpts_raw = np.array(all_kpts)          # (N_frames, 17, 3)
time_all = np.arange(len(kpts_raw)) / float(fps)

np.savez(
    npz_out,
    kpts_raw=kpts_raw,
    head_xy=np.array(all_head_xy),
    head_source=np.array(all_head_source),
    fps=float(fps),
    time_all=time_all,
    state=np.array(all_state),
    person1_idx=np.array(all_idx),
)

print(f"[OK] NPZ saved: {npz_out}")


# ----------------------------------------------------------
# 8. Re-encode for macOS QuickTime compatibility
#
# OpenCV-written MP4 can be problematic on QuickTime (seeking, playback).
# Re-encoding with ffmpeg (H.264, yuv420p, baseline) improves compatibility.
# ----------------------------------------------------------
os.system(
    f"ffmpeg -y -i {out_try_path} "
    f"-vcodec libx264 "
    f"-pix_fmt yuv420p "
    f"-profile:v baseline -level 3.0 "
    f"-movflags +faststart "
    f"{out_qt_path}"
)

print(f"[OK] QuickTime-friendly video saved: {out_qt_path}")
