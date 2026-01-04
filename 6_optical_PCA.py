# ==========================================================
# PCA (PC1) for Body-axis ROI Optical Flow
#   - fixed PC1 (全区間で1回PCA)
#   - dynamic PC1 (sliding-window PCAで軸を追跡)
#
# 重要な設計:
#   - 軸推定(PCA)は窓内中心化でOK
#   - pc1_dyn の射影は中心化しない (v(t)・e1(t)) -> 振幅指標にも安定
#
# 出力:
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
# Style
# --------------------------
rcParams["font.family"] = "DejaVu Sans"
rcParams["font.size"] = 10

# --------------------------
# I/O
# --------------------------
flow_csv = "/content/flow.csv"
out_dir  = "/content"
os.makedirs(out_dir, exist_ok=True)
SAVE_DPI = 100

# --------------------------
# Preprocess params
# --------------------------
T0_SEC              = 0.0
AUTO_T0_MIN_RUN_SEC = 0.5
BPF_LOW_HZ          = 0.5
BPF_HIGH_HZ         = 5.0
BPF_ORDER           = 4
MIN_SAMPLES_PCA     = 3

# --------------------------
# Dynamic PC1 params (METHOD準拠)
# --------------------------
USE_DYNAMIC_PC1  = True

WIN_SEC          = 2.0     # method: 2.0 s
STEP_SEC         = 0.1     # method: 0.1 s
AXIS_SMOOTH_SEC  = 0.3     # 0でもOK, ただし0.2-0.3は見た目が安定しやすい
PROJECT_CENTERED = False   # method: v(t)·e1(t) を推奨

# ==========================================================
# Utility functions
# ==========================================================
def butter_bandpass_sos(low, high, fs, order=4):
    """Butterworth band-pass filter (SOS形式)"""
    return butter(order,
                  [low/(0.5*fs), high/(0.5*fs)],
                  btype="band",
                  output="sos")

def sos_required_padlen(sos):
    """sosfiltfilt に必要な最小データ長を計算"""
    nsec = sos.shape[0]
    ntaps = 2 * nsec + 1
    return 3 * (ntaps - 1)

def finite_runs(mask):
    """True が連続する区間 [(start, end), ...] を返す"""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    gap = np.where(np.diff(idx) > 1)[0]
    s = np.r_[idx[0], idx[gap + 1]]
    e = np.r_[idx[gap], idx[-1]]
    return list(zip(s, e))

def bandpass_nanrobust(x, sos):
    """
    NaN を含む時系列に対して安全に band-pass filter を適用
    ・連続して finite な区間だけを filtfilt
    ・短すぎる区間は無視
    """
    x = np.asarray(x, float)
    y = np.full_like(x, np.nan)

    m = np.isfinite(x)
    minlen = sos_required_padlen(sos) + 1

    for s, e in finite_runs(m):
        seg = x[s:e+1]
        if len(seg) < minlen:
            continue

        pad = min(sos_required_padlen(sos), len(seg)//2 - 1)
        if pad <= 0:
            from scipy.signal import sosfilt
            y[s:e+1] = sosfilt(sos, seg)
        else:
            y[s:e+1] = sosfiltfilt(sos, seg, padlen=pad)

    return y

def auto_t0(time, vx, vy, base_t0, need_sec):
    """vx, vy が両方 finite な区間が need_sec 秒以上連続する最初の時刻を返す"""
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
    """全区間でPCAし, 固定軸PC1を返す（中心化して射影）"""
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
    return pc, w, m.sum(), mu

def align_axis_to_ref(w, ref=np.array([0.0, 1.0])):
    """軸wの向きを ref（vy正方向）へ揃える"""
    if np.any(~np.isfinite(w)):
        return w
    return -w if (np.dot(w, ref) < 0) else w

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_on()
    ax.grid(True, alpha=0.25)

def moving_average_nan(x, win):
    """NaN許容の移動平均（winはサンプル数）"""
    x = np.asarray(x, float)
    if win <= 1:
        return x.copy()
    y = np.full_like(x, np.nan)
    for i in range(len(x)):
        s = max(0, i - win//2)
        e = min(len(x), i + win//2 + 1)
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
    sliding-window PCAで軸e1(t)を推定し, v(t)を射影して pc1_dyn を作る.

    軸推定: 窓内中心化 (vx,vy の窓平均を引く) -> 共分散推定が安定.
    射影:
      - project_centered=False (推奨): pc1 = v(t)·e1(t)
      - project_centered=True         : pc1 = (v(t)-窓平均)·e1(t)

    重要:
      - PCA軸は正負が任意なので, e1_y >= 0 となるように向きを統一する.
      - さらに, 連続窓で e1 と -e1 が混在しないように, prev_w との内積で符号整合する.
      - 角度平滑化後も, 必ず e1_y >= 0 を再強制する.
    """
    n = len(time)
    pc1_dyn = np.full(n, np.nan)
    e1_x = np.full(n, np.nan)
    e1_y = np.full(n, np.nan)

    if n < MIN_SAMPLES_PCA:
        return pc1_dyn, e1_x, e1_y

    fs = 1.0 / np.median(np.diff(time))
    win_n = max(3, int(round(win_sec * fs)))
    step_n = max(1, int(round(step_sec * fs)))

    prev_w = None
    centers, W_list, MU_list = [], [], []

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

        # 1) y方向を頭側(+)に統一, dot(w, ref) < 0 なら反転.
        w = align_axis_to_ref(w, ref=ref)

        # 2) 連続窓で e1 と -e1 が混在しないよう符号整合.
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
    W = np.vstack(W_list)   # (K, 2)
    MU = np.vstack(MU_list) # (K, 2)

    # 各時刻に最も近い中心窓を割り当て.
    idx_near = np.searchsorted(centers, np.arange(n), side="left")
    idx_near = np.clip(idx_near, 0, len(centers) - 1)

    pick = np.zeros(n, dtype=int)
    for i in range(n):
        j = idx_near[i]
        j2 = max(0, j - 1)
        pick[i] = j2 if abs(i - centers[j2]) < abs(i - centers[j]) else j

    e1_x = W[pick, 0]
    e1_y = W[pick, 1]

    # 軸角度の平滑化（角度でunwrapしてから）.
    if axis_smooth_sec and axis_smooth_sec > 0:
        ang = np.arctan2(e1_y, e1_x)
        ang_u = np.unwrap(ang)
        win_ang = max(1, int(round(axis_smooth_sec * fs)))
        ang_u_sm = moving_average_nan(ang_u, win_ang)

        e1_x = np.cos(ang_u_sm)
        e1_y = np.sin(ang_u_sm)

        # ★重要: 平滑化後も, y>=0 を再強制（methodと一致）.
        flip = np.isfinite(e1_y) & (e1_y < 0)
        e1_x[flip] *= -1
        e1_y[flip] *= -1

    # 射影.
    m_all = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(e1_x) & np.isfinite(e1_y)
    if project_centered:
        mu_t = MU[pick]  # 各時刻に割り当てられた窓平均
        pc1_dyn[m_all] = (vx[m_all] - mu_t[m_all, 0]) * e1_x[m_all] + (vy[m_all] - mu_t[m_all, 1]) * e1_y[m_all]
    else:
        pc1_dyn[m_all] = vx[m_all] * e1_x[m_all] + vy[m_all] * e1_y[m_all]

    return pc1_dyn, e1_x, e1_y

# ==========================================================
# Load
# ==========================================================
df = pd.read_csv(flow_csv)
time_all = df["t_sec"].to_numpy(float)
vx_raw   = df["vx_body"].to_numpy(float)
vy_raw   = df["vy_body"].to_numpy(float)

fps = 1.0 / np.median(np.diff(time_all))

# ==========================================================
# Band-pass
# ==========================================================
sos = butter_bandpass_sos(BPF_LOW_HZ, BPF_HIGH_HZ, fps, BPF_ORDER)
vx_f = bandpass_nanrobust(vx_raw, sos)
vy_f = bandpass_nanrobust(vy_raw, sos)

# ==========================================================
# Auto t0 and trim
# ==========================================================
T0_adj = auto_t0(time_all, vx_f, vy_f, T0_SEC, AUTO_T0_MIN_RUN_SEC)
mask = time_all >= T0_adj
time = time_all[mask]
vx_f = vx_f[mask]
vy_f = vy_f[mask]

# ==========================================================
# Fixed PC1
# ==========================================================
pc1_fixed, w_fixed, n_used, mu_fixed = pca_pc1_fixed(vx_f, vy_f)
w_fixed = align_axis_to_ref(w_fixed, ref=np.array([0.0, 1.0]))
if np.isfinite(w_fixed).all():
    m = np.isfinite(vx_f) & np.isfinite(vy_f)
    X = np.column_stack([vx_f[m], vy_f[m]])
    Xc = X - X.mean(axis=0)
    pc1_fixed = np.full_like(vx_f, np.nan)
    pc1_fixed[m] = Xc @ w_fixed

# ==========================================================
# Dynamic PC1
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

e1_angle_deg = np.degrees(np.arctan2(e1y_dyn, e1x_dyn))

# ==========================================================
# Plot
# ==========================================================
fig, axs = plt.subplots(3, 1, figsize=(7, 8), dpi=SAVE_DPI, sharex=True)

axs[0].plot(time, vx_f, label="vx_body (BPF)")
axs[0].plot(time, vy_f, label="vy_body (BPF)")
axs[0].set_title("Body-axis ROI Flow (Band-pass filtered)")
axs[0].legend()
clean_axes(axs[0])

axs[1].plot(time, pc1_fixed, label="PC1 fixed")
axs[1].set_title("PC1 fixed (single PCA on entire segment)")
axs[1].set_ylabel("amplitude")
axs[1].legend()
clean_axes(axs[1])

axs[2].plot(time, pc1_dyn, label=f"PC1 dynamic (WIN={WIN_SEC}s, STEP={STEP_SEC}s)")
axs[2].set_title("PC1 dynamic (sliding-window PCA)")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("amplitude")
axs[2].legend()
clean_axes(axs[2])

fig.tight_layout()
png_path = os.path.join(out_dir, "flow_pc1.png")
fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
plt.show()

# ==========================================================
# Save CSV
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

print("Saved:")
print(png_path)
print(out_csv2)
