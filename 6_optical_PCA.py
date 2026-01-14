# ==========================================================
# PCA (PC1) for Body-axis ROI Optical Flow
#
# Goal
#   flow.csv contains ROI-mean optical flow in a body coordinate system:
#     - vx_body: mean flow along the horizontal body axis (ex)
#     - vy_body: mean flow along the vertical body axis (ey)
#
# In this cell we:
#   1) band-pass filter vx, vy (0.5–5 Hz) to suppress drift and high-frequency noise
#   2) estimate the dominant direction in the vx–vy plane using PCA
#   3) project (vx, vy) onto that direction to obtain a 1D waveform (PC1)
#
# Two PC1 variants are implemented:
#   A) Fixed PC1:
#      - run PCA once on the entire segment
#      - keep the principal axis fixed
#      - pc1_fixed is projection onto the fixed axis (centered projection)
#
#   B) Dynamic PC1 (sliding-window PCA):
#      - run PCA in short windows
#      - the principal axis e1(t) can change over time
#      - pc1_dyn is projection onto e1(t)
#
# Design rationale (important)
#   - For estimating the PCA axis within a window, centering (subtracting the mean) is appropriate,
#     because PCA is about covariance and variance directions.
#   - For dynamic PC1, we recommend non-centered projection:
#       pc1_dyn(t) = v(t) · e1(t)
#     which tends to behave more stably as an amplitude-related feature.
#
# Outputs
#   /content/flow_pc1.png
#   /content/flow_pc1.csv
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import butter, sosfiltfilt


# --------------------------
# Plot styling
# --------------------------
rcParams["font.family"] = "DejaVu Sans"
rcParams["font.size"] = 10


# --------------------------
# Input / output
# --------------------------
flow_csv = "/content/flow.csv"   # produced by the optical-flow cell
out_dir  = "/content"
os.makedirs(out_dir, exist_ok=True)
SAVE_DPI = 300


# --------------------------
# Preprocessing parameters
# --------------------------
T0_SEC              = 0.0   # analysis start time (seconds)
AUTO_T0_MIN_RUN_SEC = 0.5   # require continuous valid vx/vy for this duration
BPF_LOW_HZ          = 0.5   # band-pass low cutoff
BPF_HIGH_HZ         = 5.0   # band-pass high cutoff
BPF_ORDER           = 4     # Butterworth filter order
MIN_SAMPLES_PCA     = 3     # minimal samples for PCA (safety)


# --------------------------
# Dynamic PC1 parameters (methods-compatible)
# --------------------------
USE_DYNAMIC_PC1  = True

WIN_SEC          = 2.0     # window length in seconds
STEP_SEC         = 0.1     # step in seconds
AXIS_SMOOTH_SEC  = 0.3     # smoothing of axis angle (0 = no smoothing)
PROJECT_CENTERED = False   # recommended: False (non-centered projection)


# ==========================================================
# Utility functions
# ==========================================================
def butter_bandpass_sos(low, high, fs, order=4):
    """
    Create a Butterworth band-pass filter in SOS form.

    SOS (second-order sections) is numerically stable for long signals.
    """
    nyq = 0.5 * fs
    return butter(
        order,
        [low / nyq, high / nyq],
        btype="band",
        output="sos"
    )


def sos_required_padlen(sos):
    """
    Estimate the minimum required length for sosfiltfilt padding.

    filtfilt needs padding at both ends. If a segment is too short,
    filtering can fail or become unstable.
    """
    nsec = sos.shape[0]
    ntaps = 2 * nsec + 1
    return 3 * (ntaps - 1)


def finite_runs(mask):
    """
    Return contiguous runs of True as [(start, end), ...] (inclusive).

    This is used to filter only contiguous finite segments without
    interpolating across NaNs.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    gap = np.where(np.diff(idx) > 1)[0]
    s = np.r_[idx[0], idx[gap + 1]]
    e = np.r_[idx[gap], idx[-1]]
    return list(zip(s, e))


def bandpass_nanrobust(x, sos):
    """
    Apply band-pass filtering while safely handling NaNs.

    Strategy
      - Identify contiguous segments of finite values.
      - Filter only segments long enough for sosfiltfilt.
      - Leave short segments as NaN (avoid creating artificial signals).
    """
    x = np.asarray(x, float)
    y = np.full_like(x, np.nan)

    m = np.isfinite(x)
    minlen = sos_required_padlen(sos) + 1

    for s, e in finite_runs(m):
        seg = x[s:e + 1]
        if len(seg) < minlen:
            continue

        pad = min(sos_required_padlen(sos), len(seg) // 2 - 1)
        if pad <= 0:
            from scipy.signal import sosfilt
            y[s:e + 1] = sosfilt(sos, seg)
        else:
            y[s:e + 1] = sosfiltfilt(sos, seg, padlen=pad)

    return y


def auto_t0(time, vx, vy, base_t0, need_sec):
    """
    Find the earliest time when both vx and vy remain finite for >= need_sec.

    This is useful because the earliest part can be unstable (e.g., first frame
    has no optical flow, axes may be invalid, etc.).
    """
    ok = np.isfinite(vx) & np.isfinite(vy)
    idx = np.where(ok)[0]
    if len(idx) < MIN_SAMPLES_PCA:
        return base_t0

    gap = np.where(np.diff(idx) > 1)[0]
    s = np.r_[idx[0], idx[gap + 1]]
    e = np.r_[idx[gap], idx[-1]]

    for si, ei in zip(s, e):
        if (time[ei] - time[si]) >= need_sec:
            return max(base_t0, float(time[si]))

    return base_t0


def pca_pc1_fixed(vx, vy):
    """
    Compute a fixed PC1 by running PCA once on the entire segment.

    We center the data (subtract the mean) to estimate covariance and
    then project centered data onto the first eigenvector.

    Note: the PCA axis direction is ambiguous (w and -w are equivalent),
    so we align the sign later.
    """
    m = np.isfinite(vx) & np.isfinite(vy)
    if m.sum() < MIN_SAMPLES_PCA:
        return np.full_like(vx, np.nan), np.array([np.nan, np.nan]), 0, np.array([np.nan, np.nan])

    X = np.column_stack([vx[m], vy[m]])

    mu = X.mean(axis=0)
    Xc = X - mu

    C = np.cov(Xc, rowvar=False)
    wvals, V = np.linalg.eigh(C)
    w = V[:, np.argmax(wvals)]

    pc = np.full_like(vx, np.nan)
    pc[m] = Xc @ w
    return pc, w, int(m.sum()), mu


def align_axis_to_ref(w, ref=np.array([0.0, 1.0])):
    """
    Align the PCA axis sign to a reference direction.

    PCA gives w and -w as the same solution. Without alignment,
    PC1 signs can flip across subjects.

    If dot(w, ref) < 0, we flip w -> -w.
    """
    if np.any(~np.isfinite(w)):
        return w
    return -w if (np.dot(w, ref) < 0) else w


def clean_axes(ax):
    """Make plots cleaner for publication-style figures."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_on()
    ax.grid(True, alpha=0.25)


def moving_average_nan(x, win):
    """
    NaN-tolerant moving average.

    win is the window length in samples.
    """
    x = np.asarray(x, float)
    if win <= 1:
        return x.copy()

    y = np.full_like(x, np.nan)
    for i in range(len(x)):
        s = max(0, i - win // 2)
        e = min(len(x), i + win // 2 + 1)
        seg = x[s:e]
        if np.isfinite(seg).sum() == 0:
            continue
        y[i] = np.nanmean(seg)
    return y


def dynamic_pc1_sliding(time, vx, vy, win_sec, step_sec,
                        axis_smooth_sec=0.0,
                        ref=np.array([0.0, 1.0]),
                        project_centered=False):
    """
    Sliding-window PCA to estimate a time-varying axis e1(t) and compute pc1_dyn.

    High-level steps
      1) Slide a window along time.
      2) Within each window, estimate PCA axis (centered, covariance-based).
      3) Align axis sign:
         - first align to ref (e.g., positive ey direction)
         - then align to previous window to keep continuity
      4) Assign the nearest window axis to each time point.
      5) Compute:
           pc1_dyn(t) = v(t) · e1(t)                (recommended)
         or centered projection if project_centered=True.

    Returns
    -------
    pc1_dyn : ndarray (N,)
    e1_x, e1_y : ndarray (N,)
        Components of the time-varying PCA axis.
    """
    n = len(time)
    pc1_dyn = np.full(n, np.nan)
    e1_x = np.full(n, np.nan)
    e1_y = np.full(n, np.nan)

    if n < MIN_SAMPLES_PCA:
        return pc1_dyn, e1_x, e1_y

    # Estimate sampling rate from median time step.
    fs = 1.0 / np.median(np.diff(time))

    # Convert window/step duration to samples.
    win_n = max(3, int(round(win_sec * fs)))
    step_n = max(1, int(round(step_sec * fs)))

    prev_w = None
    centers, W_list, MU_list = [], [], []

    # --- Collect PCA axes from sliding windows ---
    for start in range(0, n - win_n + 1, step_n):
        end = start + win_n

        vx_seg = vx[start:end]
        vy_seg = vy[start:end]

        m = np.isfinite(vx_seg) & np.isfinite(vy_seg)
        if m.sum() < MIN_SAMPLES_PCA:
            continue

        X = np.column_stack([vx_seg[m], vy_seg[m]])
        mu = X.mean(axis=0)
        Xc = X - mu

        C = np.cov(Xc, rowvar=False)
        wvals, V = np.linalg.eigh(C)
        w = V[:, np.argmax(wvals)]

        # Align to reference direction
        w = align_axis_to_ref(w, ref=ref)

        # Align to previous window direction for continuity
        if prev_w is not None and np.dot(w, prev_w) < 0:
            w = -w
        prev_w = w.copy()

        c = (start + end - 1) // 2
        centers.append(c)
        W_list.append(w)
        MU_list.append(mu)

    if len(centers) == 0:
        return pc1_dyn, e1_x, e1_y

    centers = np.array(centers, int)
    W = np.vstack(W_list)    # (K,2)
    MU = np.vstack(MU_list)  # (K,2)

    # --- Assign the nearest window axis to each sample index ---
    idx_near = np.searchsorted(centers, np.arange(n), side="left")
    idx_near = np.clip(idx_near, 0, len(centers) - 1)

    pick = np.zeros(n, dtype=int)
    for i in range(n):
        j = idx_near[i]
        j2 = max(0, j - 1)
        pick[i] = j2 if abs(i - centers[j2]) < abs(i - centers[j]) else j

    e1_x = W[pick, 0]
    e1_y = W[pick, 1]

    # --- Optional smoothing of axis angle over time ---
    if axis_smooth_sec and axis_smooth_sec > 0:
        ang = np.arctan2(e1_y, e1_x)
        ang_u = np.unwrap(ang)
        win_ang = max(1, int(round(axis_smooth_sec * fs)))
        ang_u_sm = moving_average_nan(ang_u, win_ang)

        e1_x = np.cos(ang_u_sm)
        e1_y = np.sin(ang_u_sm)

        # Keep y >= 0 to reduce sign flips after smoothing
        flip = np.isfinite(e1_y) & (e1_y < 0)
        e1_x[flip] *= -1
        e1_y[flip] *= -1

    # --- Projection to obtain pc1_dyn ---
    m_all = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(e1_x) & np.isfinite(e1_y)

    if project_centered:
        mu_t = MU[pick]
        pc1_dyn[m_all] = (vx[m_all] - mu_t[m_all, 0]) * e1_x[m_all] + (vy[m_all] - mu_t[m_all, 1]) * e1_y[m_all]
    else:
        pc1_dyn[m_all] = vx[m_all] * e1_x[m_all] + vy[m_all] * e1_y[m_all]

    return pc1_dyn, e1_x, e1_y


# ==========================================================
# Load flow.csv
# ==========================================================
df = pd.read_csv(flow_csv)

time_all = df["t_sec"].to_numpy(float)
vx_raw   = df["vx_body"].to_numpy(float)
vy_raw   = df["vy_body"].to_numpy(float)

# Estimate fps from time spacing
fps = 1.0 / np.median(np.diff(time_all))


# ==========================================================
# Band-pass filter (0.5–5 Hz)
# ==========================================================
sos = butter_bandpass_sos(BPF_LOW_HZ, BPF_HIGH_HZ, fps, BPF_ORDER)

vx_f = bandpass_nanrobust(vx_raw, sos)
vy_f = bandpass_nanrobust(vy_raw, sos)


# ==========================================================
# Auto t0 and trim (skip initial unstable part)
# ==========================================================
T0_adj = auto_t0(time_all, vx_f, vy_f, T0_SEC, AUTO_T0_MIN_RUN_SEC)

mask = time_all >= T0_adj
time = time_all[mask]
vx_f = vx_f[mask]
vy_f = vy_f[mask]


# ==========================================================
# Fixed PC1 (single PCA on the entire segment)
# ==========================================================
pc1_fixed, w_fixed, n_used, mu_fixed = pca_pc1_fixed(vx_f, vy_f)

# Align axis sign so that it points toward +ey direction (vy-positive).
w_fixed = align_axis_to_ref(w_fixed, ref=np.array([0.0, 1.0]))

# Recompute projection with aligned axis (centered projection for fixed PC1)
if np.isfinite(w_fixed).all():
    m = np.isfinite(vx_f) & np.isfinite(vy_f)
    X = np.column_stack([vx_f[m], vy_f[m]])
    Xc = X - X.mean(axis=0)

    pc1_fixed = np.full_like(vx_f, np.nan)
    pc1_fixed[m] = Xc @ w_fixed


# ==========================================================
# Dynamic PC1 (sliding-window PCA)
# ==========================================================
if USE_DYNAMIC_PC1:
    pc1_dyn, e1x_dyn, e1y_dyn = dynamic_pc1_sliding(
        time=time,
        vx=vx_f,
        vy=vy_f,
        win_sec=WIN_SEC,
        step_sec=STEP_SEC,
        axis_smooth_sec=AXIS_SMOOTH_SEC,
        ref=np.array([0.0, 1.0]),
        project_centered=PROJECT_CENTERED,
    )
else:
    pc1_dyn = np.full_like(vx_f, np.nan)
    e1x_dyn = np.full_like(vx_f, np.nan)
    e1y_dyn = np.full_like(vx_f, np.nan)

# Axis angle in degrees (useful for QC)
e1_angle_deg = np.degrees(np.arctan2(e1y_dyn, e1x_dyn))


# ==========================================================
# Plot
# ==========================================================
fig, axs = plt.subplots(3, 1, figsize=(7, 8), dpi=SAVE_DPI, sharex=True)

# Panel 1: filtered vx, vy
axs[0].plot(time, vx_f, label="vx_body (BPF)")
axs[0].plot(time, vy_f, label="vy_body (BPF)")
axs[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
axs[0].set_title("Body-axis ROI Flow (Band-pass filtered)")
clean_axes(axs[0])

# Panel 2: fixed PC1
axs[1].plot(time, pc1_fixed, label="PC1 fixed")
axs[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
axs[1].set_title("PC1 fixed (single PCA on entire segment)")
axs[1].set_ylabel("Amplitude (a.u.)")
clean_axes(axs[1])

# Panel 3: dynamic PC1
axs[2].plot(time, pc1_dyn, label=f"PC1 dynamic (WIN={WIN_SEC}s, STEP={STEP_SEC}s)")
axs[2].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
axs[2].set_title("PC1 dynamic (sliding-window PCA)")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Amplitude (a.u.)")
clean_axes(axs[2])

fig.tight_layout()
fig.subplots_adjust(right=0.78)

png_path = os.path.join(out_dir, "flow_pc1.png")
fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
plt.show()


# ==========================================================
# Save CSV for downstream analysis
# ==========================================================
out_csv2 = os.path.join(out_dir, "flow_pc1.csv")

pd.DataFrame({
    "t_sec": time,
    "vx_body_bpf": vx_f,
    "vy_body_bpf": vy_f,
    "pc1_fixed": pc1_fixed,
    "pc1_dyn": pc1_dyn,
    "e1_dyn_x": e1x_dyn,
    "e1_dyn_y": e1y_dyn,
    "e1_dyn_angle_deg": e1_angle_deg,
}).to_csv(out_csv2, index=False)

print("Saved.")
print(png_path)
print(out_csv2)
