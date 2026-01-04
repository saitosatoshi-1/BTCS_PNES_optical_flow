import numpy as np
import cv2
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.ndimage import uniform_filter1d
import os

# ==========================================================
# Flow PC1 waveform overlay (time-synced with upstream cell)
# ==========================================================

# ----------------------------------------------------------
# upstream と同じ解析区間を指定
# ----------------------------------------------------------
#START_SEC = 0.0     # 上流セルと同じ値にする
#END_SEC   = None    # None なら最後まで

# ----------------------------------------------------------
# 入力
# ----------------------------------------------------------
video_path = "/content/flow.mp4"
csv_path   = "/content/flow_pc1.csv"

out_raw = "/content/flow_pc1_raw.mp4"
out_fin = "/content/flow_pc1_qt.mp4"

# ----------------------------------------------------------
# Plot A と同じ平滑化
# ----------------------------------------------------------
SMOOTH_SEC = 0.20
plot_w = 450


# ==========================================================
# utility: NaN-robust moving average
# ==========================================================
def ensure_odd(n):
    return int(n) | 1

def smooth_ma(x, fs, sec):
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


# ==========================================================
# PC1 読み込み & 区間切り出し
# ==========================================================
df = pd.read_csv(csv_path)

t_all  = df["t_sec"].to_numpy(float)
pc1_all = df["pc1_dyn"].to_numpy(float)

mask = np.isfinite(pc1_all) & (t_all >= START_SEC)
if END_SEC is not None:
    mask &= (t_all <= END_SEC)

t_raw = t_all[mask]
pc1   = pc1_all[mask]

if t_raw.size < 2:
    raise RuntimeError("PC1 time series too short")

# ---- 0 秒基準に再基準化（上流セルと一致） ----
t = t_raw - t_raw[0]
t_end = float(t[-1])

# sampling rate 推定
dt = np.diff(t)
dt = dt[np.isfinite(dt) & (dt > 0)]
fs_wave = 1.0 / np.median(dt)

# 平滑化（Plot A と同一）
pc1_s = smooth_ma(pc1, fs_wave, SMOOTH_SEC)


# ==========================================================
# 動画オープン
# ==========================================================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Video open failed")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ---- 解析開始フレームへジャンプ ----
start_frame = int(round(START_SEC * fps))
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)


# ==========================================================
# 出力動画
# ==========================================================
out = cv2.VideoWriter(
    out_raw,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W + plot_w, H)
)
if not out.isOpened():
    cap.release()
    raise RuntimeError("VideoWriter failed")


# ==========================================================
# 波形 Figure（1 回だけ作成）
# ==========================================================
dpi = 100
fig_w_in = plot_w / dpi
fig_h_in = H / dpi

fig = Figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
fig.patch.set_facecolor("white")

canvas = FigureCanvasAgg(fig)
ax = fig.add_subplot(111)

ax.set_xlabel("Time (sec)")
ax.set_ylabel("PC1 (flow)")

ax.plot(t, pc1_s, linewidth=1)
ax.set_xlim(0, t_end)

y = pc1_s[np.isfinite(pc1_s)]
if y.size >= 10:
    ylo, yhi = np.percentile(y, [1, 99])
    if yhi > ylo:
        pad = 0.10 * (yhi - ylo)
        ax.set_ylim(ylo - pad, yhi + pad)

ax.grid(True, alpha=0.3)

vline = ax.axvline(0.0, linewidth=2, color="red")

fig.tight_layout()
canvas.draw()


# ==========================================================
# メインループ
# ==========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- 解析開始からの経過時間（0 秒基準） ----
    t_cur = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 - START_SEC
    if t_cur < 0 or t_cur > t_end:
        break

    vline.set_xdata([t_cur, t_cur])

    canvas.draw()
    plot_rgb = np.asarray(canvas.buffer_rgba())[..., :3]
    plot_bgr = cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR)

    if plot_bgr.shape[0] != H or plot_bgr.shape[1] != plot_w:
        plot_bgr = cv2.resize(plot_bgr, (plot_w, H))

    combined = np.hstack([frame, plot_bgr])
    out.write(combined)

cap.release()
out.release()


# ==========================================================
# QuickTime 用再エンコード
# ==========================================================
os.system(
    f"ffmpeg -y -i {out_raw} -vcodec libx264 -pix_fmt yuv420p "
    f"-profile:v baseline -level 3.0 -movflags +faststart {out_fin}"
)

print("=== DONE ===")
print("[OUTPUT]", out_fin)
