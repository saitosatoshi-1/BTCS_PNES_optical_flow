"""
pca_dynamic_pc1_core.py

Purpose
-------
Compute the dynamic first principal component waveform (pc1_dyn) from 2D body-axis
optical-flow features (vx_body, vy_body) using sliding-window PCA.

Method (matches manuscript description)
--------------------------------------
Given v(t) = [vx(t), vy(t)]:

1) Band-pass filter vx and vy (default 0.5–5 Hz) using a zero-phase SOS filter.
   Filtering is applied only to contiguous finite runs (NaNs are not interpolated).

2) Sliding-window PCA:
   - Window length: WIN_SEC
   - Step size: STEP_SEC
   - For each window, compute the covariance matrix of v(t) within that window
     (after mean-centering for covariance estimation).
   - Define e1 as the eigenvector corresponding to the largest eigenvalue.

3) Dynamic PC1 projection:
     pc1_dyn(t) = v(t) · e1(t)
   where e1(t) is assigned to each time sample by the nearest window center.
   (This is a non-centered projection, consistent with the manuscript statement.)

Input
-----
flow.csv with columns: t_sec, vx_body, vy_body

Output
------
flow_pc1.csv with columns: t_sec, pc1_dyn
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


# =========================
# Parameters
# =========================
FLOW_CSV = "flow.csv"
OUT_CSV = "flow_pc1.csv"

fs = 30
BPF_LOW_HZ = 0.5
BPF_HIGH_HZ = 5.0
BPF_ORDER = 4

WIN_SEC = 2.0
STEP_SEC = 0.1

MIN_SAMPLES_PCA = 3


# =========================
# Utilities: time and filtering
# =========================
def butter_bandpass_sos(low_hz: float, high_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Create a Butterworth band-pass filter in SOS form (numerically stable).
    """
    nyq = 0.5 * fs
    if not (0 < low_hz < high_hz < nyq):
        raise ValueError(f"Invalid band-pass range. low={low_hz}, high={high_hz}, nyquist={nyq}.")
    return butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")


def sos_required_padlen(sos: np.ndarray) -> int:
    """
    Conservative minimum length for stable sosfiltfilt padding.
    """
    nsec = int(sos.shape[0])
    ntaps = 2 * nsec + 1
    return 3 * (ntaps - 1)


def finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Return contiguous True runs of a boolean mask as (start, end) inclusive.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    gap = np.where(np.diff(idx) > 1)[0]
    s = np.r_[idx[0], idx[gap + 1]]
    e = np.r_[idx[gap], idx[-1]]
    return [(int(a), int(b)) for a, b in zip(s, e)]


def bandpass_nanrobust(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """
    Zero-phase band-pass filtering without interpolating across NaNs.

    Only contiguous finite segments long enough for sosfiltfilt are filtered.
    Short segments are left as NaN to avoid creating artificial dynamics.
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
            # Extremely short edge case: keep segment unfiltered rather than unstable filtering.
            y[s:e + 1] = seg
        else:
            y[s:e + 1] = sosfiltfilt(sos, seg, padlen=pad)

    return y


# =========================
# Utilities: PCA axis handling and dynamic PC1
# =========================
def align_axis_to_ref(w: np.ndarray, ref: np.ndarray = np.array([0.0, 1.0])) -> np.ndarray:
    """
    Resolve eigenvector sign ambiguity by enforcing dot(w, ref) >= 0.
    """
    if np.any(~np.isfinite(w)):
        return w
    return -w if float(np.dot(w, ref)) < 0 else w


def dynamic_pc1_sliding(
    time_sec: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    win_sec: float,
    step_sec: float,
    ref: np.ndarray = np.array([0.0, 1.0]),
) -> np.ndarray:
    """
    Compute dynamic PC1 using sliding-window PCA.

    Implementation details
    ----------------------
    - Window-level axis estimation:
        * Select finite samples in the window.
        * Mean-center within the window to compute covariance.
        * Use eigenvector of the largest eigenvalue as e1.
    - Axis assignment:
        * Store axes at window centers.
        * For each sample index, assign the axis from the nearest window center.
    - Projection (non-centered, manuscript-consistent):
        pc1_dyn(t) = v(t) · e1(t)

    Returns
    -------
    pc1_dyn : ndarray (N,)
        Dynamic PC1 waveform, NaN where not computable.
    """
    time_sec = np.asarray(time_sec, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)

    n = int(time_sec.size)
    pc1_dyn = np.full(n, np.nan)

    if n < MIN_SAMPLES_PCA:
        return pc1_dyn

    win_n = max(MIN_SAMPLES_PCA, int(round(win_sec * fs)))
    step_n = max(1, int(round(step_sec * fs)))

    centers: list[int] = []
    W_list: list[np.ndarray] = []
    prev_w = None

    for start in range(0, n - win_n + 1, step_n):
        end = start + win_n

        vx_seg = vx[start:end]
        vy_seg = vy[start:end]

        m = np.isfinite(vx_seg) & np.isfinite(vy_seg)
        if int(m.sum()) < MIN_SAMPLES_PCA:
            continue

        # Build 2D velocity samples for PCA.
        X = np.column_stack([vx_seg[m], vy_seg[m]])

        # Mean-centering is used ONLY for covariance estimation.
        Xc = X - X.mean(axis=0)

        C = np.cov(Xc, rowvar=False)
        vals, V = np.linalg.eigh(C)
        w = V[:, int(np.argmax(vals))]  # first eigenvector (largest eigenvalue)

        # Sign alignment to reduce arbitrary flips.
        w = align_axis_to_ref(w, ref=ref)
        if prev_w is not None and float(np.dot(w, prev_w)) < 0:
            w = -w
        prev_w = w.copy()

        c = int((start + end - 1) // 2)
        centers.append(c)
        W_list.append(w)

    if len(centers) == 0:
        return pc1_dyn

    centers_arr = np.asarray(centers, dtype=int)
    W = np.vstack(W_list)  # (K, 2)

    # Assign the nearest window center to each sample index.
    idx_near = np.searchsorted(centers_arr, np.arange(n), side="left")
    idx_near = np.clip(idx_near, 0, len(centers_arr) - 1)

    pick = np.zeros(n, dtype=int)
    for i in range(n):
        j = int(idx_near[i])
        j2 = max(0, j - 1)
        pick[i] = j2 if abs(i - int(centers_arr[j2])) < abs(i - int(centers_arr[j])) else j

    e1_x = W[pick, 0]
    e1_y = W[pick, 1]

    # Non-centered projection (manuscript-consistent):
    #   pc1_dyn(t) = [vx(t), vy(t)] · [e1_x(t), e1_y(t)]
    m_all = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(e1_x) & np.isfinite(e1_y)
    pc1_dyn[m_all] = vx[m_all] * e1_x[m_all] + vy[m_all] * e1_y[m_all]

    return pc1_dyn


# =========================
# Main
# =========================
def main() -> None:
    df = pd.read_csv(FLOW_CSV)

    required = {"t_sec", "vx_body", "vy_body"}
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing columns in {FLOW_CSV}. Required={sorted(list(required))}, missing={missing}.")

    t = df["t_sec"].to_numpy(float)
    vx = df["vx_body"].to_numpy(float)
    vy = df["vy_body"].to_numpy(float)


    # Band-pass filtering (NaN-safe, no interpolation).
    sos = butter_bandpass_sos(BPF_LOW_HZ, BPF_HIGH_HZ, fs, order=BPF_ORDER)
    vx_bpf = bandpass_nanrobust(vx, sos)
    vy_bpf = bandpass_nanrobust(vy, sos)

    # Dynamic PC1 (sliding-window PCA + non-centered projection).
    pc1_dyn = dynamic_pc1_sliding(
        time_sec=t,
        vx=vx_bpf,
        vy=vy_bpf,
        win_sec=WIN_SEC,
        step_sec=STEP_SEC,
        ref=np.array([0.0, 1.0]),
    )

    # Save only what is needed downstream.
    pd.DataFrame({"t_sec": t, "pc1_dyn": pc1_dyn}).to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()
