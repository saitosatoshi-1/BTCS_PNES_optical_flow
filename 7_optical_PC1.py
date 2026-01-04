# ==========================================================
# PC1 Panels (dynamic PC1 only, publication-ready)
#   - flow_pc1.csv の pc1_dyn だけを使って解析する
#   - 出力: flow_3PANEL_dynamic.png, flow_summary_dynamic.csv
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress, kendalltau
from matplotlib import rcParams

# --------------------------
# Style
# --------------------------
rcParams["font.family"] = "DejaVu Sans"
rcParams["font.size"] = 10

# --------------------------
# I/O
# --------------------------
in_csv  = "/content/flow_pc1.csv"
out_dir = "/content"
os.makedirs(out_dir, exist_ok=True)

out_png = os.path.join(out_dir, "flow_3PANEL_dyn.png")
out_csv = os.path.join(out_dir, "flow_summary_dyn.csv")

# --------------------------
# Parameters
# --------------------------
START_SEC  = 2.5
END_SEC    = 12.5   # 10 s window
SMOOTH_SEC = 0.20

# flow_pc1.csv のどの列を使うか（dynamicのみ）
PC1_COL = "pc1_dyn"

# --------------------------
# Utility functions
# --------------------------
def ensure_odd(n: int) -> int:
    """移動平均窓を奇数にする（中心揃えのため）"""
    return int(n) | 1

def smooth_ma(x, fs, sec):
    """
    NaNに頑健な移動平均
    - uniform_filter1d を使って高速に平均
    - NaNは0として足し合わせ, 有効サンプル数で割る
    """
    if sec <= 0:
        return np.asarray(x, float)

    k = ensure_odd(max(1, int(round(fs * sec))))
    x = np.asarray(x, float)

    valid = np.isfinite(x).astype(float)
    x2 = x.copy()
    x2[~np.isfinite(x2)] = 0.0

    num = uniform_filter1d(x2, size=k, mode="nearest")
    den = uniform_filter1d(valid, size=k, mode="nearest")

    y = num / np.maximum(den, 1e-9)
    y[den < 1e-9] = np.nan
    return y

def estimate_fps(t):
    """時刻列からfpsを推定（微小なズレ対策）"""
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return 1.0 / np.median(dt) if dt.size else 30.0

def regression_line(time, y):
    """
    ln(y) = intercept + slope*time の回帰をして, 指数近似曲線yhatも返す
    - yは正である必要があるので, logの前に下限を入れる
    """
    m = np.isfinite(time) & np.isfinite(y)
    t = time[m]
    v = y[m]
    if v.size < 5:
        return dict(slope=np.nan, r=np.nan, yhat=np.full_like(time, np.nan))

    yy = np.log(np.maximum(v, 1e-9))
    slope, intercept, r, _, _ = linregress(t, yy)
    return dict(slope=slope, r=r, yhat=np.exp(intercept + slope * time))

def detect_cycles(pc1, time, fs,
                  peak_min_frac=0.20,   # ★追加: 小ピーク抑制, 0.15-0.30あたりで調整
                  peak_min_abs=0.0,     # ★追加: 絶対値しきい値, 必要なら >0 にする
                  min_dist_sec=0.2):
    """
    周期検出（微小ピーク抑制つき）
    1) PC1をSMOOTH_SECで平滑化
    2) 0上向きクロッシング→次の0下向きクロッシングを1周期窓
    3) その窓の正側最大をピーク候補
    4) ★ピーク高さが小さい場合は捨てる
    5) 0.2秒以内の近接ピークは統合（最大振幅のものを残す）
    """
    pc1_s = smooth_ma(pc1, fs, SMOOTH_SEC)

    # ★全体スケールからピークの最低高さを決める（正側だけで評価）
    pos = pc1_s[np.isfinite(pc1_s) & (pc1_s > 0)]
    if pos.size >= 10:
        ref = np.nanpercentile(pos, 95)  # 発作中の「十分大きい振幅」の代表
        peak_thr = max(peak_min_abs, peak_min_frac * ref)
    else:
        peak_thr = peak_min_abs

    up = np.where((pc1_s[:-1] <= 0) & (pc1_s[1:] > 0))[0]
    dn = np.where((pc1_s[:-1] > 0) & (pc1_s[1:] <= 0))[0]

    t_raw, a_raw = [], []
    for iu in up:
        dn_after = dn[dn > iu]
        if dn_after.size == 0:
            continue

        seg = pc1_s[iu:dn_after[0] + 1]
        if seg.size == 0 or np.all(~np.isfinite(seg)):
            continue

        im = int(np.nanargmax(seg))
        a_peak = float(seg[im])

        # ★追加: 微小ピークを捨てる
        if not np.isfinite(a_peak) or a_peak < peak_thr:
            continue

        t_raw.append(time[iu + im])
        a_raw.append(a_peak)

    if len(t_raw) < 2:
        return pc1_s, np.asarray(t_raw), np.array([]), np.array([])

    t_raw = np.asarray(t_raw)
    a_raw = np.asarray(a_raw)

    # 近接ピーク統合
    MIN_DIST = float(min_dist_sec)
    t_keep = [t_raw[0]]
    a_keep = [a_raw[0]]

    for t, a in zip(t_raw[1:], a_raw[1:]):
        if t - t_keep[-1] < MIN_DIST:
            if a > a_keep[-1]:
                t_keep[-1] = t
                a_keep[-1] = a
        else:
            t_keep.append(t)
            a_keep.append(a)

    t_peaks = np.asarray(t_keep)
    if t_peaks.size < 2:
        return pc1_s, t_peaks, np.array([]), np.array([])

    T  = np.diff(t_peaks)
    tm = 0.5 * (t_peaks[:-1] + t_peaks[1:])
    ok = np.isfinite(T) & (T > 0)

    return pc1_s, t_peaks, tm[ok], T[ok]

# --------------------------
# Load & cut (dynamic PC1 only)
# --------------------------
df = pd.read_csv(in_csv)

# 必須列チェック（なくても落ちないように明示）
if PC1_COL not in df.columns:
    raise KeyError(f"Column '{PC1_COL}' not found in {in_csv}. Available: {list(df.columns)}")

t_all   = df["t_sec"].to_numpy(float)
pc1_all = df[PC1_COL].to_numpy(float)

# 解析窓を切り出し（NaNは除く）
mask = (np.isfinite(pc1_all) &
        (t_all >= START_SEC) &
        (t_all <= END_SEC))

t_raw = t_all[mask]
pc1   = pc1_all[mask]

# timeを0開始に正規化
time = t_raw - t_raw[0]
fs = estimate_fps(time)

# ==========================================================
# Feature extraction (ALL metrics calculated ONCE)
# ==========================================================

# --- amplitude metrics ---
# |PC1| を平滑化して振幅系列とする
amp = smooth_ma(np.abs(pc1), fs, SMOOTH_SEC)

# ADS（ln(amp)の回帰傾き）
ads = regression_line(time, amp)
ADS_R2_all = ads["r"]**2 if np.isfinite(ads["r"]) else np.nan

# time masks for early/late
m0_5  = (time >= 0) & (time <= 5)
m5_10 = (time >= 5) & (time < 10)

# ADS by segment
ads_0_5  = regression_line(time[m0_5],  amp[m0_5])
ads_5_10 = regression_line(time[m5_10], amp[m5_10])

ADS_slope_0_5  = ads_0_5["slope"]
ADS_slope_5_10 = ads_5_10["slope"]

# AUC（振幅の時間積分）
area_all = np.trapz(amp, time)
area_0_5 = np.trapz(amp[m0_5],  time[m0_5])  if m0_5.sum()  >= 2 else np.nan
area_5_10 = np.trapz(amp[m5_10], time[m5_10]) if m5_10.sum() >= 2 else np.nan
area_ratio_5_10_0_5 = area_5_10 / area_0_5 if (np.isfinite(area_0_5) and area_0_5 > 0) else np.nan

# mean |PC1|
M_0_5  = np.nanmean(amp[m0_5])
M_5_10 = np.nanmean(amp[m5_10])
R_5_10_0_5 = M_5_10 / M_0_5 if (np.isfinite(M_0_5) and M_0_5 > 0) else np.nan

# --- cycles / frequency metrics ---
pc1_s, t_peaks, tm, T = detect_cycles(pc1, time, fs)
f = 1.0 / np.maximum(T, 1e-9)

tm0_5  = (tm >= 0) & (tm < 5)
tm5_10 = (tm >= 5) & (tm < 10)

f_early = np.nanmean(f[tm0_5]) if tm0_5.sum() >= 2 else np.nan
f_late  = np.nanmean(f[tm5_10]) if tm5_10.sum() >= 2 else np.nan
f_ratio = f_late / f_early if np.isfinite(f_early) else np.nan

coef_all = np.polyfit(tm, f, 1) if f.size >= 3 else None

# --- periodicity (Kendall tau) ---
kendall_tau_all, kendall_p_all = kendalltau(tm, T) if tm.size >= 5 else (np.nan, np.nan)
kendall_tau_0_5, kendall_p_0_5 = kendalltau(tm[tm0_5], T[tm0_5]) if tm0_5.sum() >= 4 else (np.nan, np.nan)
kendall_tau_5_10, kendall_p_5_10 = kendalltau(tm[tm5_10], T[tm5_10]) if tm5_10.sum() >= 4 else (np.nan, np.nan)

# CV of IJI（補助指標）
iji_cv_all = np.nanstd(T) / np.nanmean(T) if T.size >= 3 else np.nan
iji_cv_0_5 = np.nanstd(T[tm0_5]) / np.nanmean(T[tm0_5]) if tm0_5.sum() >= 3 else np.nan
iji_cv_5_10 = np.nanstd(T[tm5_10]) / np.nanmean(T[tm5_10]) if tm5_10.sum() >= 3 else np.nan

# ==========================================================
# Plot (3 panels)
# ==========================================================
fig = plt.figure(figsize=(6, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.35)

axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1], sharex=axA)
axC = fig.add_subplot(gs[2], sharex=axA)

# (A) PC1 waveform + peaks
axA.plot(time, pc1_s, label=f"{PC1_COL} (smoothed)")
axA.scatter(t_peaks, np.interp(t_peaks, time, pc1_s), s=32, edgecolors="k", facecolors="C1")
axA.set_ylabel("PC1 (a.u.)")
axA.set_title("(A) Dynamic PC1 waveform and positive peaks")
axA.grid(True, alpha=0.3)
axA.text(
    0.02, 0.05,
    f"Peaks = {len(t_peaks)}\n"
    f"τ all = {kendall_tau_all:.2f}\n"
    f"τ 0–5 = {kendall_tau_0_5:.2f}, τ 5–10 = {kendall_tau_5_10:.2f}",
    transform=axA.transAxes, va="bottom",
    bbox=dict(boxstyle="round", fc="white")
)

# (B) Frequency change
axB.scatter(tm[tm0_5], f[tm0_5], label="0–5 s")
axB.scatter(tm[tm5_10], f[tm5_10], label="5–10 s")
if coef_all is not None:
    tline = np.array([tm.min(), tm.max()])
    axB.plot(tline, np.polyval(coef_all, tline), "k--", label="linear fit (all)")
axB.set_ylabel("Frequency (Hz)")
axB.set_title("(B) Frequency change (from dynamic PC1 peaks)")
axB.legend()
axB.grid(True, alpha=0.3)
axB.text(
    0.02, 0.95,
    f"mean f 0–5 = {f_early:.2f}\n"
    f"mean f 5–10 = {f_late:.2f}\n"
    f"ratio = {f_ratio:.2f}\n"
    f"slope all = {(coef_all[0] if coef_all is not None else np.nan):.3f}",
    transform=axB.transAxes, va="top",
    bbox=dict(boxstyle="round", fc="white")
)

# (C) Amplitude change
axC.plot(time[m0_5],  amp[m0_5],  label="|PC1| 0–5 s")
axC.plot(time[m5_10], amp[m5_10], label="|PC1| 5–10 s")
axC.plot(time, ads["yhat"], "k--", label="ADS fit (all)")
axC.set_ylabel("|PC1| (a.u.)")
axC.set_xlabel("Time (s)")
axC.set_title("(C) Amplitude change (from dynamic PC1)")
axC.legend()
axC.grid(True, alpha=0.3)
axC.text(
    0.02, 0.95,
    f"mean_0_5 = {M_0_5:.2f}\n"
    f"mean_5_10 = {M_5_10:.2f}\n"
    f"area_0_5 = {area_0_5:.2f}\n"
    f"area_5_10 = {area_5_10:.2f}\n",
    transform=axC.transAxes, va="top",
    bbox=dict(boxstyle="round", fc="white")
)
axC.text(
    0.62, 0.95,
    f"ADS all = {ads['slope']:.3f}\n"
    f"ADS 0–5 = {ADS_slope_0_5:.3f}\n"
    f"ADS 5–10 = {ADS_slope_5_10:.3f}",
    transform=axC.transAxes, va="top",
    bbox=dict(boxstyle="round", fc="white")
)

fig.tight_layout()
fig.savefig(out_png, dpi=100, bbox_inches="tight")
plt.show()

# ==========================================================
# Final CSV (ALL features, dynamic-only)
# ==========================================================
pd.DataFrame([{
    "PC1_source": PC1_COL,
    "Peak_n": len(t_peaks),

    "PC1_mean_0_5": M_0_5,
    "PC1_mean_5_10": M_5_10,
    "PC1_mean_ratio": R_5_10_0_5,

    "PC1_area_all": area_all,
    "PC1_area_0_5": area_0_5,
    "PC1_area_5_10": area_5_10,
    "PC1_area_ratio": area_ratio_5_10_0_5,

    "ADS_slope_all": ads["slope"],
    "ADS_R2_all": ADS_R2_all,
    "ADS_slope_0_5": ADS_slope_0_5,
    "ADS_slope_5_10": ADS_slope_5_10,

    "Freq_mean_0_5": f_early,
    "Freq_mean_5_10": f_late,
    "Freq_ratio": f_ratio,
    "Freq_slope_all": coef_all[0] if coef_all is not None else np.nan,

    "Kendall_tau_all": kendall_tau_all,
    "Kendall_tau_0_5": kendall_tau_0_5,
    "Kendall_tau_5_10": kendall_tau_5_10,

    "IJI_CV_all": iji_cv_all,
    "IJI_CV_0_5": iji_cv_0_5,
    "IJI_CV_5_10": iji_cv_5_10,
}]).to_csv(out_csv, index=False)

print("[OK] Figure saved:", out_png)
print("[OK] CSV saved:", out_csv)
