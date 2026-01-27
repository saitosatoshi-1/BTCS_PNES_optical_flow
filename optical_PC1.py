"""
pc1_dynamic_metrics_core.py

Purpose
-------
Compute three dynamic-PC1 metrics from flow_pc1.csv using only pc1_dyn:
  1) PC1 area (AUC of |PC1|), 0–10 s
  2) ADS (Amplitude Decay Slope): slope of ln(|PC1|) vs time, 0–10 s
  3) Kendall tau: monotonic trend of inter-peak interval T vs time, 0–10 s

Input
-----
flow_pc1.csv with required columns:
  - t_sec
  - pc1_dyn

Output
------
flow_summary_dyn_core.csv (single-row table)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress, kendalltau


# =========================
# Parameters
# =========================
IN_CSV = "flow_pc1.csv"
OUT_CSV = "flow_summary_dyn_core.csv"

PC1_COL = "pc1_dyn"

WINDOW_SEC = 10.0          # analyze 0–10 s after the first valid sample
SMOOTH_SEC = 0.20          # smoothing for |PC1| and peak detection stability

# Peak detection thresholds (positive peaks only).
PEAK_MIN_FRAC = 0.20       # fraction of a robust amplitude reference (95th percentile)
PEAK_MIN_ABS = 0.0         # absolute threshold (use >0 only if you need hard gating)
MIN_DIST_SEC = 0.2         # merge peaks closer than this


def ensure_odd(n: int) -> int:
    """
    Ensure an integer is odd.
    An odd window size makes the moving window symmetric around the center sample.
    """
    return int(n) | 1


def smooth_ma_nan(x: np.ndarray, fs: float, sec: float) -> np.ndarray:
    """
    NaN-tolerant moving average.

    Idea.
      - Replace NaNs with 0 for the numerator.
      - Count valid samples for the denominator.
      - Divide numerator by denominator, keep NaN where denominator is 0.

    This avoids interpolating across missing data.
    """
    x = np.asarray(x, dtype=float)

    if sec <= 0:
        return x.copy()

    k = ensure_odd(max(1, int(round(fs * sec))))

    valid = np.isfinite(x).astype(float)
    x2 = x.copy()
    x2[~np.isfinite(x2)] = 0.0

    num = uniform_filter1d(x2, size=k, mode="nearest")
    den = uniform_filter1d(valid, size=k, mode="nearest")

    y = num / np.maximum(den, 1e-12)
    y[den < 1e-12] = np.nan
    return y


def rolling_p95_positive(pc1_s: np.ndarray, fs: float, win_sec: float) -> np.ndarray:
    """
    Compute a rolling 95th percentile of the positive PC1 amplitude.

    Method requirement.
      - Apply a 2.0-s moving window to the positive amplitude of PC1.
      - Compute the 95th percentile within that window.
      - Use it as a local amplitude reference for peak-thresholding.

    Implementation.
      - Use a centered window of length win_sec.
      - Only positive values (pc1_s > 0) contribute.
      - NaNs are ignored.
      - If a window has too few valid points, output NaN for that index.

    Notes.
      - For a 10-s segment at ~30 fps, this loop is fast enough and keeps behavior explicit.
      - If you later process very long recordings, we can optimize this.
    """
    pc1_s = np.asarray(pc1_s, dtype=float)

    win_n = int(round(win_sec * fs))
    win_n = max(3, ensure_odd(win_n))
    half = win_n // 2

    # Keep only positive amplitude, everything else becomes NaN.
    pos = pc1_s.copy()
    pos[~np.isfinite(pos)] = np.nan
    pos[pos <= 0] = np.nan

    p95 = np.full(pos.shape, np.nan, dtype=float)

    for i in range(pos.size):
        s = max(0, i - half)
        e = min(pos.size, i + half + 1)

        seg = pos[s:e]
        seg = seg[np.isfinite(seg)]
        if seg.size < 5:
            continue

        p95[i] = float(np.percentile(seg, 95))

    return p95


def detect_cycles_positive_peaks(
    pc1: np.ndarray,
    time_sec: np.ndarray,
    fs: float,
    smooth_sec: float = 0.20,
    p95_win_sec: float = 2.0,
    peak_min_frac: float = 0.20,
    peak_min_abs: float = 0.0,
    min_dist_sec: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect positive peaks and derive inter-peak intervals, aligned to your Methods.

    Steps.
      1) Smooth PC1 with a 0.2-s moving average.
      2) Define each cycle window as:
           upward zero crossing (y = 0) -> subsequent downward zero crossing.
      3) Within each cycle window, take the maximum value as the peak.
      4) Suppress over-detection using a local threshold:
           local_ref(t) = rolling 95th percentile of positive PC1 amplitude
                          in a 2.0-s moving window.
           keep peak only if:
             peak >= max(peak_min_abs, peak_min_frac * local_ref(at peak time)).
      5) Merge peaks that are closer than 0.2 s by keeping the larger one.

    Returns.
      pc1_s  : smoothed PC1
      t_peaks: peak times
      tm     : midpoints between successive peaks
      T      : inter-peak intervals (seconds)
    """
    pc1 = np.asarray(pc1, dtype=float)
    time_sec = np.asarray(time_sec, dtype=float)

    # 1) Smooth PC1 for stable zero-crossing and cycle segmentation.
    pc1_s = smooth_ma_nan(pc1, fs, smooth_sec)

    # 2) Compute the local 95th percentile reference (2.0-s moving window).
    local_p95 = rolling_p95_positive(pc1_s, fs, win_sec=p95_win_sec)

    # 3) Zero-crossings on the smoothed waveform.
    up = np.where((pc1_s[:-1] <= 0) & (pc1_s[1:] > 0))[0]
    dn = np.where((pc1_s[:-1] > 0) & (pc1_s[1:] <= 0))[0]

    t_raw: list[float] = []
    a_raw: list[float] = []

    # 4) For each upward crossing, find the next downward crossing, then pick the max.
    for iu in up:
        dn_after = dn[dn > iu]
        if dn_after.size == 0:
            continue

        end = int(dn_after[0])
        seg = pc1_s[iu:end + 1]

        if seg.size == 0 or np.all(~np.isfinite(seg)):
            continue

        im = int(np.nanargmax(seg))
        ipk = int(iu + im)

        a_peak = float(seg[im])
        if not np.isfinite(a_peak):
            continue

        # Local threshold at the peak index.
        ref = float(local_p95[ipk]) if np.isfinite(local_p95[ipk]) else np.nan
        thr = float(peak_min_abs)

        if np.isfinite(ref) and ref > 0:
            thr = max(thr, float(peak_min_frac) * ref)

        # Reject minor peaks below the local threshold.
        if a_peak < thr:
            continue

        t_raw.append(float(time_sec[ipk]))
        a_raw.append(a_peak)

    if len(t_raw) < 2:
        return pc1_s, np.asarray(t_raw, float), np.array([]), np.array([])

    t_raw = np.asarray(t_raw, float)
    a_raw = np.asarray(a_raw, float)

    # 5) Merge peaks closer than min_dist_sec by keeping the largest amplitude.
    t_keep = [float(t_raw[0])]
    a_keep = [float(a_raw[0])]

    for t, a in zip(t_raw[1:], a_raw[1:]):
        if float(t) - float(t_keep[-1]) < float(min_dist_sec):
            if float(a) > float(a_keep[-1]):
                t_keep[-1] = float(t)
                a_keep[-1] = float(a)
        else:
            t_keep.append(float(t))
            a_keep.append(float(a))

    t_peaks = np.asarray(t_keep, float)
    if t_peaks.size < 2:
        return pc1_s, t_peaks, np.array([]), np.array([])

    T = np.diff(t_peaks)
    tm = 0.5 * (t_peaks[:-1] + t_peaks[1:])

    ok = np.isfinite(T) & (T > 0)
    return pc1_s, t_peaks, tm[ok], T[ok]


# =========================
# Main
# =========================
df = pd.read_csv(IN_CSV)

required = {"t_sec", PC1_COL}
missing = [c for c in required if c not in df.columns]
if len(missing) > 0:
    raise KeyError(f"Missing columns in {IN_CSV}. Required={sorted(list(required))}, missing={missing}.")

t_all = df["t_sec"].to_numpy(float)
pc1_all = df[PC1_COL].to_numpy(float)

# Keep only finite pairs.
m = np.isfinite(t_all) & np.isfinite(pc1_all)
t_all = t_all[m]
pc1_all = pc1_all[m]

if t_all.size < 10:
    raise RuntimeError("Too few valid samples in input CSV.")

# Define the analysis window as 0–10 s from the first valid time point.
t0 = float(t_all[0])
time = t_all - t0

m_win = (time >= 0.0) & (time <= float(WINDOW_SEC))
time = time[m_win]
pc1 = pc1_all[m_win]

if time.size < 10:
    raise RuntimeError("Too few samples in the 0–10 s window.")

fs = estimate_fs_from_time(time)

# --- Metric 1: PC1 area (AUC of |PC1|) ---
amp = smooth_ma_nan(np.abs(pc1), fs, SMOOTH_SEC)
pc1_area_0_10 = safe_auc(amp, time)

# --- Metric 2: ADS (slope of ln(|PC1|) vs time) ---
ads = exp_decay_regression(time, amp)
ads_slope_0_10 = float(ads["slope"])
ads_r2_0_10 = float(ads["r"] ** 2) if np.isfinite(ads["r"]) else float("nan")

# --- Metric 3: Kendall tau of inter-peak intervals ---
_, t_peaks, tm, T = detect_cycles_positive_peaks(pc1, time, fs)
if tm.size >= 5:
    tau, p = kendalltau(tm, T)
    kendall_tau_0_10 = float(tau)
    kendall_p_0_10 = float(p)
else:
    kendall_tau_0_10 = float("nan")
    kendall_p_0_10 = float("nan")

# Save a single-row summary CSV.
summary = pd.DataFrame(
    [
        {
            "PC1_source": PC1_COL,
            "window_sec": float(WINDOW_SEC),
            "PC1_area_0_10": float(pc1_area_0_10),
            "ADS_slope_0_10": float(ads_slope_0_10),
            "ADS_R2_0_10": float(ads_r2_0_10),
            "Kendall_tau_0_10": float(kendall_tau_0_10),
            "Kendall_p_0_10": float(kendall_p_0_10),
            "Peak_n": int(t_peaks.size),
        }
    ]
)
summary.to_csv(OUT_CSV, index=False)
