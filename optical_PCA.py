"""
pca_dynamic_pc1_beginner.py

Goal.
  flow.csv の vx_body, vy_body から, dynamic PC1(pc1_dyn) を作る.

Background.
  - vx_body, vy_body は body座標系の ROI平均 optical flow.
  - 2次元(vx, vy) を 1次元に要約したい.
  - PCAの第1主成分方向 e1(t) を sliding window で推定し,
    pc1_dyn(t) = v(t) · e1(t) (non-centered projection) を出力する.

Input.
  flow.csv with columns:
    - t_sec
    - vx_body
    - vy_body

Output.
  - flow_pc1.csv: t_sec, vx_body_bpf, vy_body_bpf, pc1_dyn, e1_dyn_x, e1_dyn_y
  - flow_pc1_dyn.png: QC plot (vx,vy and pc1_dyn)

Notes.
  - Band-pass filtering is applied to suppress drift and high-frequency noise.
  - NaNs are handled without interpolation, filtering is applied only to valid contiguous runs.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams
from scipy.signal import butter, sosfiltfilt


# ==========================================================
# Style.
# ==========================================================
rcParams["font.family"] = "DejaVu Sans"
rcParams["font.size"] = 10


# ==========================================================
# I/O.
# ==========================================================
FLOW_CSV = "flow.csv"
OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "flow_pc1.csv")
OUT_PNG = os.path.join(OUT_DIR, "flow_pc1_dyn.png")


# ==========================================================
# Parameters.
# ==========================================================
BPF_LOW_HZ = 0.5
BPF_HIGH_HZ = 5.0
BPF_ORDER = 4

MIN_SAMPLES_PCA = 3

WIN_SEC = 2.0
STEP_SEC = 0.1
AXIS_SMOOTH_SEC = 0.3

PROJECT_CENTERED = False  # recommended, False


# ==========================================================
# Small helper functions.
# ==========================================================
def butter_bandpass_sos(low_hz: float, high_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """Create a Butterworth band-pass filter in SOS form."""
    nyq = 0.5 * fs
    if not (0 < low_hz < high_hz < nyq):
        raise ValueError(f"Invalid band-pass range. low={low_hz}, high={high_hz}, nyquist={nyq}.")
    return butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")


def sos_required_padlen(sos: np.ndarray) -> int:
    """
    Estimate minimum length for stable sosfiltfilt padding.
    This is a conservative heuristic.
    """
    nsec = int(sos.shape[0])
    ntaps = 2 * nsec + 1
    return 3 * (ntaps - 1)


def finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of True as [(start, end), ...] (inclusive)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    gap = np.where(np.diff(idx) > 1)[0]
    s = np.r_[idx[0], idx[gap + 1]]
    e = np.r_[idx[gap], idx[-1]]
    return [(int(a), int(b)) for a, b in zip(s, e)]


def bandpass_nanrobust(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """
    Band-pass filter that does not interpolate across NaNs.
    Only long-enough contiguous finite segments are filtered.
    """
    x = np.asarray(x, dtype=float)
    y = np.full_like(x, np.nan)

    m = np.isfinite(x)
    minlen = sos_required_padlen(sos) + 1

    for s, e in finite_runs(m):
        seg = x[s:e + 1]
        if seg.size < minlen:
            continue

        pad = min(sos_required_padlen(sos), int(seg.size // 2 - 1))
        if pad <= 0:
            y[s:e + 1] = seg
        else:
            y[s:e + 1] = sosfiltfilt(sos, seg, padlen=pad)

    return y


def align_axis_to_ref(w: np.ndarray, ref: np.ndarray = np.array([0.0, 1.0])) -> np.ndarray:
    """
    PCA axis has sign ambiguity, w and -w are equivalent.
    Align w so that dot(w, ref) >= 0.
    """
    if np.any(~np.isfinite(w)):
        return w
    return -w if float(np.dot(w, ref)) < 0 else w


def moving_average_nan(x: np.ndarray, win: int) -> np.ndarray:
    """NaN-tolerant moving average, simple and beginner-friendly."""
    x = np.asarray(x, dtype=float)
    if win <= 1:
        return x.copy()

    y = np.full_like(x, np.nan)
    half = int(win // 2)

    for i in range(x.size):
        s = max(0, i - half)
        e = min(x.size, i + half + 1)
        seg = x[s:e]
        if np.isfinite(seg).sum() == 0:
            continue
        y[i] = float(np.nanmean(seg))

    return y


def dynamic_pc1_sliding(
    time_sec: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    win_sec: float,
    step_sec: float,
    axis_smooth_sec: float = 0.0,
    ref: np.ndarray = np.array([0.0, 1.0]),
    project_centered: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window PCA to obtain a time-varying axis e1(t) and dynamic PC1.

    Returns.
      pc1_dyn, e1_x, e1_y
    """
    time_sec = np.asarray(time_sec, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)

    n = int(time_sec.size)
    pc1_dyn = np.full(n, np.nan)
    e1_x = np.full(n, np.nan)
    e1_y = np.full(n, np.nan)

    if n < MIN_SAMPLES_PCA:
        return pc1_dyn, e1_x, e1_y

    dt = np.diff(time_sec)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    fs = float(1.0 / np.median(dt)) if dt.size else 30.0

    win_n = max(3, int(round(win_sec * fs)))
    step_n = max(1, int(round(step_sec * fs)))

    centers = []
    W_list = []
    MU_list = []

    prev_w = None

    for start in range(0, n - win_n + 1, step_n):
        end = start + win_n

        vx_seg = vx[start:end]
        vy_seg = vy[start:end]
        m = np.isfinite(vx_seg) & np.isfinite(vy_seg)

        if int(m.sum()) < MIN_SAMPLES_PCA:
            continue

        X = np.column_stack([vx_seg[m], vy_seg[m]])
        mu = X.mean(axis=0)
        Xc = X - mu

        C = np.cov(Xc, rowvar=False)
        vals, V = np.linalg.eigh(C)
        w = V[:, int(np.argmax(vals))]

        w = align_axis_to_ref(w, ref=ref)

        if prev_w is not None and float(np.dot(w, prev_w)) < 0:
            w = -w
        prev_w = w.copy()

        c = int((start + end - 1) // 2)
        centers.append(c)
        W_list.append(w)
        MU_list.append(mu)

    if len(centers) == 0:
        return pc1_dyn, e1_x, e1_y

    centers = np.asarray(centers, dtype=int)
    W = np.vstack(W_list)
    MU = np.vstack(MU_list)

    # Assign the nearest window axis to each sample.
    idx_near = np.searchsorted(centers, np.arange(n), side="left")
    idx_near = np.clip(idx_near, 0, len(centers) - 1)

    pick = np.zeros(n, dtype=int)
    for i in range(n):
        j = int(idx_near[i])
        j2 = max(0, j - 1)
        pick[i] = j2 if abs(i - int(centers[j2])) < abs(i - int(centers[j])) else j

    e1_x = W[pick, 0]
    e1_y = W[pick, 1]

    # Optional smoothing of axis angle.
    if axis_smooth_sec and axis_smooth_sec > 0:
        ang = np.arctan2(e1_y, e1_x)
        ang_u = np.unwrap(ang)

        win_ang = max(1, int(round(axis_smooth_sec * fs)))
        ang_u_sm = moving_average_nan(ang_u, win_ang)

        e1_x = np.cos(ang_u_sm)
        e1_y = np.sin(ang_u_sm)

        flip = np.isfinite(e1_y) & (e1_y < 0)
        e1_x[flip] *= -1
        e1_y[flip] *= -1

    # Projection.
    m_all = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(e1_x) & np.isfinite(e1_y)

    if project_centered:
        mu_t = MU[pick]
        pc1_dyn[m_all] = (vx[m_all] - mu_t[m_all, 0]) * e1_x[m_all] + (vy[m_all] - mu_t[m_all, 1]) * e1_y[m_all]
    else:
        pc1_dyn[m_all] = vx[m_all] * e1_x[m_all] + vy[m_all] * e1_y[m_all]

    return pc1_dyn, e1_x, e1_y


def clean_axes(ax) -> None:
    """Simple publication-style axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)


# ==========================================================
# Main.
# ==========================================================
df = pd.read_csv(FLOW_CSV)

need_cols = {"t_sec", "vx_body", "vy_body"}
missing = [c for c in need_cols if c not in df.columns]
if len(missing) > 0:
    raise KeyError(f"Missing columns in {FLOW_CSV}. Required: {sorted(list(need_cols))}. Missing: {missing}.")

time_all = df["t_sec"].to_numpy(float)
vx_raw = df["vx_body"].to_numpy(float)
vy_raw = df["vy_body"].to_numpy(float)

dt = np.diff(time_all)
dt = dt[np.isfinite(dt) & (dt > 0)]
fps = float(1.0 / np.median(dt)) if dt.size else 30.0

sos = butter_bandpass_sos(BPF_LOW_HZ, BPF_HIGH_HZ, fps, order=BPF_ORDER)
vx_bpf = bandpass_nanrobust(vx_raw, sos)
vy_bpf = bandpass_nanrobust(vy_raw, sos)

pc1_dyn, e1x, e1y = dynamic_pc1_sliding(
    time_sec=time_all,
    vx=vx_bpf,
    vy=vy_bpf,
    win_sec=WIN_SEC,
    step_sec=STEP_SEC,
    axis_smooth_sec=AXIS_SMOOTH_SEC,
    ref=np.array([0.0, 1.0]),
    project_centered=PROJECT_CENTERED,
)

# Save CSV.
out = pd.DataFrame(
    {
        "t_sec": time_all,
        "vx_body_bpf": vx_bpf,
        "vy_body_bpf": vy_bpf,
        "pc1_dyn": pc1_dyn,
        "e1_dyn_x": e1x,
        "e1_dyn_y": e1y,
    }
)
out.to_csv(OUT_CSV, index=False)

# QC plot.
fig, axs = plt.subplots(2, 1, figsize=(7, 6), dpi=200, sharex=True)

axs[0].plot(time_all, vx_bpf, label="vx_body (BPF)")
axs[0].plot(time_all, vy_bpf, label="vy_body (BPF)")
axs[0].set_title("Body-axis ROI flow, band-pass filtered")
axs[0].set_ylabel("Flow (a.u.)")
axs[0].legend()
clean_axes(axs[0])

axs[1].plot(time_all, pc1_dyn, label=f"pc1_dyn (WIN={WIN_SEC}s, STEP={STEP_SEC}s)")
axs[1].set_title("Dynamic PC1 (sliding-window PCA)")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("PC1 (a.u.)")
axs[1].legend()
clean_axes(axs[1])

fig.tight_layout()
fig.savefig(OUT_PNG, bbox_inches="tight")
plt.show()
