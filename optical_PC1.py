"""
pc1_panels_dynamic_minimal.py

目的.
  flow_pc1.csv の pc1_dyn だけを使って,
  0–10秒の区間で次の3指標だけを計算して保存する.

  1) PC1 area.
     |PC1| の時間積分(AUC), 0–10 s.

  2) ADS, Amplitude Decay Slope.
     |PC1| を指数関数で近似するために, ln(|PC1|) を time に対して回帰し,
     その傾き(slope)をADSとする, 0–10 s.
     slope < 0 なら減衰.

  3) Kendall tau.
     ピーク間隔 T が時間とともに単調に増えるかを見る指標.
     time midpoints(tm) と T の Kendall tau を計算する, 0–10 s.

入力.
  - flow_pc1.csv, 必須列.
      t_sec, pc1_dyn

出力.
  - flow_dyn_minimal.png, QC用図(2パネル)
  - flow_summary_dyn_minimal.csv, 特徴量1行
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress, kendalltau
from matplotlib import rcParams


# ==========================================================
# 1) Style
# ==========================================================
rcParams["font.family"] = "DejaVu Sans"
rcParams["font.size"] = 10


# ==========================================================
# 2) I, O
# ==========================================================
IN_CSV = "flow_pc1.csv"
OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "flow_dyn_minimal.png")
OUT_SUMMARY = os.path.join(OUT_DIR, "flow_summary_dyn_minimal.csv")


# ==========================================================
# 3) Parameters
# ==========================================================
PC1_COL = "pc1_dyn"

WINDOW_SEC = 10.0
SMOOTH_SEC = 0.20

PEAK_MIN_FRAC = 0.20
PEAK_MIN_ABS = 0.0
MIN_DIST_SEC = 0.2


# ==========================================================
# 4) Helper functions
# ==========================================================
def ensure_odd(n: int) -> int:
    """移動平均の窓を奇数にして, 中心揃えになりやすくする."""
    return int(n) | 1


def estimate_fps_from_time(t: np.ndarray) -> float:
    """時刻列からfpsを推定する, 微小なズレに強い."""
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 30.0
    return float(1.0 / np.median(dt))


def smooth_ma_nan(x: np.ndarray, fs: float, sec: float) -> np.ndarray:
    """
    NaNに強い移動平均.
    - NaNを0として足し合わせ, 有効サンプル数で割る.
    """
    x = np.asarray(x, float)

    if sec <= 0:
        return x

    k = ensure_odd(max(1, int(round(fs * sec))))

    valid = np.isfinite(x).astype(float)
    x2 = x.copy()
    x2[~np.isfinite(x2)] = 0.0

    num = uniform_filter1d(x2, size=k, mode="nearest")
    den = uniform_filter1d(valid, size=k, mode="nearest")

    y = num / np.maximum(den, 1e-9)
    y[den < 1e-9] = np.nan
    return y


def exp_decay_regression(time_sec: np.ndarray, y: np.ndarray) -> dict:
    """
    ln(y) = intercept + slope*time の回帰でADSを推定する.
    yは正が必要なので, logの前に下限を入れる.
    """
    m = np.isfinite(time_sec) & np.isfinite(y)
    t = time_sec[m]
    v = y[m]

    if v.size < 5:
        return {"slope": np.nan, "r": np.nan, "yhat": np.full_like(time_sec, np.nan)}

    vv = np.maximum(v, 1e-9)
    yy = np.log(vv)

    slope, intercept, r, _, _ = linregress(t, yy)
    yhat = np.exp(intercept + slope * time_sec)

    return {"slope": float(slope), "r": float(r), "yhat": yhat}


def detect_cycles(pc1: np.ndarray, time_sec: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    正側ピークを使って周期(ピーク間隔)を作る.

    手順.
      1) 平滑化したPC1でゼロクロッシングを探す.
      2) 上向きゼロクロッシングから次の下向きゼロクロッシングまでを1周期窓とみなし,
         その窓の最大をピークとする.
      3) 小さすぎるピークを捨てる.
      4) 近すぎるピークは統合する.

    戻り値.
      pc1_s, t_peaks, tm, T
      tmはピーク間の中点時刻, Tはピーク間隔(秒).
    """
    pc1_s = smooth_ma_nan(pc1, fs, SMOOTH_SEC)

    pos = pc1_s[np.isfinite(pc1_s) & (pc1_s > 0)]
    if pos.size >= 10:
        ref = float(np.nanpercentile(pos, 95))
        peak_thr = max(float(PEAK_MIN_ABS), float(PEAK_MIN_FRAC) * ref)
    else:
        peak_thr = float(PEAK_MIN_ABS)

    up = np.where((pc1_s[:-1] <= 0) & (pc1_s[1:] > 0))[0]
    dn = np.where((pc1_s[:-1] > 0) & (pc1_s[1:] <= 0))[0]

    t_raw = []
    a_raw = []

    for iu in up:
        dn_after = dn[dn > iu]
        if dn_after.size == 0:
            continue

        seg = pc1_s[iu:dn_after[0] + 1]
        if seg.size == 0 or np.all(~np.isfinite(seg)):
            continue

        im = int(np.nanargmax(seg))
        a_peak = float(seg[im])
        if (not np.isfinite(a_peak)) or (a_peak < peak_thr):
            continue

        t_raw.append(float(time_sec[iu + im]))
        a_raw.append(a_peak)

    if len(t_raw) < 2:
        return pc1_s, np.asarray(t_raw, float), np.array([]), np.array([])

    t_raw = np.asarray(t_raw, float)
    a_raw = np.asarray(a_raw, float)

    t_keep = [t_raw[0]]
    a_keep = [a_raw[0]]

    for t, a in zip(t_raw[1:], a_raw[1:]):
        if t - t_keep[-1] < float(MIN_DIST_SEC):
            if a > a_keep[-1]:
                t_keep[-1] = t
                a_keep[-1] = a
        else:
            t_keep.append(t)
            a_keep.append(a)

    t_peaks = np.asarray(t_keep, float)
    if t_peaks.size < 2:
        return pc1_s, t_peaks, np.array([]), np.array([])

    T = np.diff(t_peaks)
    tm = 0.5 * (t_peaks[:-1] + t_peaks[1:])

    ok = np.isfinite(T) & (T > 0)
    return pc1_s, t_peaks, tm[ok], T[ok]


def safe_auc(y: np.ndarray, t: np.ndarray) -> float:
    """有効点が2点以上あるときだけAUCを計算する."""
    m = np.isfinite(y) & np.isfinite(t)
    if int(m.sum()) < 2:
        return float("nan")
    return float(np.trapz(y[m], t[m]))


# ==========================================================
# 5) Load, cut to 0–10 s
# ==========================================================
df = pd.read_csv(IN_CSV)

if "t_sec" not in df.columns:
    raise KeyError("t_sec column is required.")

if PC1_COL not in df.columns:
    raise KeyError(f"{PC1_COL} column is required.")

t_all = df["t_sec"].to_numpy(float)
pc1_all = df[PC1_COL].to_numpy(float)

m_valid = np.isfinite(t_all) & np.isfinite(pc1_all)
t_all = t_all[m_valid]
pc1_all = pc1_all[m_valid]

if t_all.size < 10:
    raise RuntimeError("Too few valid samples in input CSV.")

# timeを0開始にして, 0–10秒を切り出す
t0 = float(t_all[0])
time = t_all - t0

m_win = (time >= 0.0) & (time <= float(WINDOW_SEC))
time = time[m_win]
pc1 = pc1_all[m_win]

if time.size < 10:
    raise RuntimeError("Too few samples in 0–10 s window. Check the input CSV.")

fs = estimate_fps_from_time(time)


# ==========================================================
# 6) Metrics, PC1 area, ADS, Kendall tau
# ==========================================================
amp = smooth_ma_nan(np.abs(pc1), fs, SMOOTH_SEC)

pc1_area_0_10 = safe_auc(amp, time)

ads = exp_decay_regression(time, amp)
ads_slope_0_10 = float(ads["slope"])
ads_r2_0_10 = float(ads["r"] ** 2) if np.isfinite(ads["r"]) else np.nan

pc1_s, t_peaks, tm, T = detect_cycles(pc1, time, fs)

if tm.size >= 5:
    kendall_tau_0_10, kendall_p_0_10 = kendalltau(tm, T)
    kendall_tau_0_10 = float(kendall_tau_0_10)
    kendall_p_0_10 = float(kendall_p_0_10)
else:
    kendall_tau_0_10, kendall_p_0_10 = np.nan, np.nan


# ==========================================================
# 7) Plot, 2 panels, minimal QC
# ==========================================================
fig = plt.figure(figsize=(6, 7))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35)

axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1], sharex=axA)

# (A) PC1 waveform + peaks
axA.plot(time, pc1_s, label="PC1 smoothed")
if t_peaks.size > 0:
    axA.scatter(t_peaks, np.interp(t_peaks, time, pc1_s), s=28, edgecolors="k", facecolors="C1")
axA.set_ylabel("PC1 (a.u.)")
axA.set_title("Dynamic PC1 waveform, 0–10 s")
axA.grid(True, alpha=0.3)
axA.text(
    0.02, 0.05,
    f"Peaks = {int(t_peaks.size)}\n"
    f"Kendall τ = {kendall_tau_0_10:.2f}",
    transform=axA.transAxes,
    va="bottom",
    bbox=dict(boxstyle="round", fc="white")
)

# (B) |PC1| and ADS fit
axB.plot(time, amp, label="|PC1| smoothed")
axB.plot(time, ads["yhat"], "k--", label="ADS fit")
axB.set_ylabel("|PC1| (a.u.)")
axB.set_xlabel("Time (s)")
axB.set_title("Amplitude, 0–10 s")
axB.grid(True, alpha=0.3)
axB.legend()
axB.text(
    0.02, 0.95,
    f"PC1 area = {pc1_area_0_10:.2f}\n"
    f"ADS slope = {ads_slope_0_10:.3f}\n"
    f"ADS R2 = {ads_r2_0_10:.2f}",
    transform=axB.transAxes,
    va="top",
    bbox=dict(boxstyle="round", fc="white")
)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.show()


# ==========================================================
# 8) Save summary CSV, one row
# ==========================================================
summary = {
    "PC1_source": PC1_COL,
    "window_sec": float(WINDOW_SEC),

    "PC1_area_0_10": float(pc1_area_0_10),

    "ADS_slope_0_10": float(ads_slope_0_10),
    "ADS_R2_0_10": float(ads_r2_0_10),

    "Kendall_tau_0_10": float(kendall_tau_0_10),
    "Kendall_p_0_10": float(kendall_p_0_10),

    "Peak_n": int(t_peaks.size),
}

pd.DataFrame([summary]).to_csv(OUT_SUMMARY, index=False)

print("Saved figure:", OUT_PNG)
print("Saved summary CSV:", OUT_SUMMARY)
