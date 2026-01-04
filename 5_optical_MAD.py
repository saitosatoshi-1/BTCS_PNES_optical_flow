# =============================================================
# Body-axis Optical Flow in Fixed ROI (manual polygon)
# QC breakdown付き
# =============================================================

import cv2
import numpy as np
import pandas as pd
from collections import deque

# =============================================================
# 入出力パス
# =============================================================
video_path = "/content/FBTCS_qt.mp4"
npz_path   = "/content/track.npz"

out_video  = "/content/flow.mp4"
out_csv    = "/content/flow.csv"
qc_csv     = "/content/flow_qc_summary.csv"

# =============================================================
# パラメータ
# =============================================================
MAD_Z_THR   = 6.0
DRAW_SPIKE  = True
WIN_SEC_MAD = 10.0

FB_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

# ----------------------------------------------------------
# 手動 BED polygon
# ----------------------------------------------------------
assert poly_rot_global is not None, "BED polygon not set"
ROI_POINTS = poly_rot_global.astype(np.float32)

# =============================================================
# Skeleton（NPZ）
# =============================================================
dat = np.load(npz_path, allow_pickle=True)
kpts     = dat["kpts_raw"]    # (T,17,3)
head_xy  = dat["head_xy"]     # (T,2)
time_all = dat["time_all"]
fps_npz  = float(dat["fps"])



# =============================================================
# head_xy の短時間欠損補間（<= 0.8 秒）
# =============================================================
head_xy_interp = head_xy.copy()

MAX_GAP_SEC = 0.8
MAX_GAP_FRAMES = int(round(MAX_GAP_SEC * fps_npz))

for d in range(2):  # x, y
    s = pd.Series(head_xy_interp[:, d])

    # 線形補間（短い欠損のみ）
    s_interp = s.interpolate(
        method="linear",
        limit=MAX_GAP_FRAMES,
        limit_direction="both"
    )

    head_xy_interp[:, d] = s_interp.values

n_before = np.sum(~np.all(np.isfinite(head_xy), axis=1))
n_after  = np.sum(~np.all(np.isfinite(head_xy_interp), axis=1))

print("\n=== head_xy interpolation QC ===")
print(f"NaN frames before : {n_before}")
print(f"NaN frames after  : {n_after}")
print(f"Max gap allowed   : {MAX_GAP_FRAMES} frames (~{MAX_GAP_SEC:.2f} sec)")





LS = 5
RS = 6

# =============================================================
# 動画
# =============================================================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"VideoCapture failed: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = fps_npz if fps_npz > 0 else 30.0

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    out_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H)
)

# =============================================================
# ROI マスク
# =============================================================
mask = np.zeros((H, W), np.uint8)
cv2.fillPoly(mask, [ROI_POINTS.astype(np.int32)], 1)
mask_bool = mask.astype(bool)

# =============================================================
# skeleton index
# =============================================================
def skeleton_index_from_t(t_sec, time_all):
    ai = int(np.searchsorted(time_all, t_sec, side="right") - 1)
    return int(np.clip(ai, 0, len(time_all) - 1))

# =============================================================
# 身体座標系（理由付き）
# =============================================================
def body_axes_from_kp(kp, head_xy_frame):
    LS_xy = kp[LS, :2]
    RS_xy = kp[RS, :2]

    # NaN チェック
    if not (np.all(np.isfinite(LS_xy)) and
            np.all(np.isfinite(RS_xy)) and
            np.all(np.isfinite(head_xy_frame))):
        return None, None, False, "nan"

    shoulder_mid = (LS_xy + RS_xy) / 2.0

    ex = RS_xy - LS_xy
    ey = head_xy_frame - shoulder_mid

    # ベクトル長チェック
    if np.linalg.norm(ex) < 1e-6 or np.linalg.norm(ey) < 1e-6:
        return None, None, False, "short"

    ex = ex / np.linalg.norm(ex)

    # Gram–Schmidt
    ey = ey - ex * np.dot(ey, ex)
    if np.linalg.norm(ey) < 1e-6:
        return None, None, False, "orth_fail"

    ey = ey / np.linalg.norm(ey)

    return ex.astype(np.float32), ey.astype(np.float32), True, "ok"

# =============================================================
# MAD バッファ
# =============================================================
mad_buf = deque(maxlen=max(20, int(round(WIN_SEC_MAD * fps))))

# =============================================================
# QC カウンタ
# =============================================================
cnt_total = 0
cnt_flow_none = 0
cnt_axes_invalid = 0
cnt_axes_nan = 0
cnt_axes_short = 0

# =============================================================
# メインループ
# =============================================================
flow_prev = None
frame_idx = 0
rows = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    t_sec = t_msec / 1000.0 if t_msec and t_msec > 0 else frame_idx / fps

    ai = skeleton_index_from_t(t_sec, time_all)
    kp = kpts[ai]
    head_pt = head_xy_interp[ai]

    cnt_total += 1

    ex, ey, axes_valid, reason = body_axes_from_kp(kp, head_pt)

    vx_mean = np.nan
    vy_mean = np.nan
    mag_mean = np.nan
    overlay = frame.copy()

    if flow_prev is None:
        cnt_flow_none += 1

    elif axes_valid:
        flow = cv2.calcOpticalFlowFarneback(flow_prev, gray, None, **FB_PARAMS)
        fx, fy = flow[...,0], flow[...,1]

        fx_body = fx * ex[0] + fy * ex[1]
        fy_body = fx * ey[0] + fy * ey[1]
        mag_body = cv2.magnitude(fx_body, fy_body)

        vx_mean  = float(np.nanmean(fx_body[mask_bool]))
        vy_mean  = float(np.nanmean(fy_body[mask_bool]))
        mag_mean = float(np.nanmean(mag_body[mask_bool]))

        ang = np.arctan2(fy_body, fx_body)
        hsv = np.zeros((H, W, 3), np.uint8)
        hsv[...,1] = 255
        hsv[...,0] = ((ang + np.pi)/(2*np.pi)*180).astype(np.uint8)
        hsv[...,2] = cv2.normalize(mag_body, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)

        flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        overlay = cv2.addWeighted(frame,0.35, flow_color*(mask_bool[...,None]),0.65,0)

    else:
        cnt_axes_invalid += 1
        if reason == "nan":
            cnt_axes_nan += 1
        else:
            cnt_axes_short += 1

    flow_prev = gray

    # MAD スパイク検出（除外しない）
    spike_flag = 0
    if np.isfinite(mag_mean):
        mad_buf.append(mag_mean)
    if len(mad_buf) >= 20 and np.isfinite(mag_mean):
        arr = np.asarray(mad_buf)
        z = abs(mag_mean - np.median(arr)) / (np.median(np.abs(arr - np.median(arr))) + 1e-9)
        if z > MAD_Z_THR:
            spike_flag = 1

    color = (0,0,255) if (spike_flag and DRAW_SPIKE) else (0,255,255)
    cv2.polylines(overlay, [ROI_POINTS.astype(int)], True, color, 2)

    out.write(overlay)
    rows.append([frame_idx, t_sec, vx_mean, vy_mean, mag_mean, spike_flag])

    frame_idx += 1

cap.release()
out.release()

# =============================================================
# CSV 保存
# =============================================================
df_out = pd.DataFrame(
    rows,
    columns=["frame","t_sec","vx_body","vy_body","mag_body","spike"]
)
df_out.to_csv(out_csv, index=False)

# =============================================================
# QC Summary
# =============================================================
mag = df_out["mag_body"].to_numpy()
valid = np.isfinite(mag)

T_total = len(valid)
T_valid = int(valid.sum())
T_excl  = T_total - T_valid
valid_rate = T_valid / T_total if T_total > 0 else np.nan

max_run = 0
cur = 0
for v in valid:
    cur = cur + 1 if not v else 0
    max_run = max(max_run, cur)

print("\n=== QC Summary ===")
print(f"Total frames            : {T_total}")
print(f"Valid frames            : {T_valid}")
print(f"Excluded frames         : {T_excl}")
print(f"Valid frame rate        : {valid_rate*100:.2f}%")
print(f"Max consecutive exclude : {max_run} frames (~{max_run/fps:.2f} sec)")

print("\n=== QC Failure Breakdown ===")
print(f"flow_prev None         : {cnt_flow_none}")
print(f"axes invalid (total)   : {cnt_axes_invalid}")
print(f"  - NaN keypoints      : {cnt_axes_nan}")
print(f"  - short/orth failure : {cnt_axes_short}")

# =============================================================
# QC CSV
# =============================================================
df_qc = pd.DataFrame([{
    "video_path": video_path,
    "npz_path": npz_path,
    "fps": fps,
    "total_frames": T_total,
    "valid_frames": T_valid,
    "excluded_frames": T_excl,
    "valid_frame_rate": valid_rate,
    "max_consecutive_exclude_frames": max_run,
    "max_consecutive_exclude_sec": max_run / fps if fps > 0 else np.nan,
    "axes_invalid": cnt_axes_invalid,
    "axes_nan": cnt_axes_nan,
    "axes_short_or_orth": cnt_axes_short,
}])

df_qc.to_csv(qc_csv, index=False)
print(f"\nQC summary saved to: {qc_csv}")
