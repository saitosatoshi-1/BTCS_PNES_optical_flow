# =============================================================
# Body-axis Optical Flow in Fixed ROI (manual polygon)
# -------------------------------------------------------------
# 1) 動画を1フレームずつ読む
# 2) 前フレームと今フレームから Optical Flow（動き）を計算する
# 3) その動きを「身体の向き（左右, 頭方向）」に合わせた座標系に変換する
# 4) 手で決めたROI（ベッド領域ポリゴン）内だけ平均して数値化する
# 5) 結果をCSVに保存する（vx_body, vy_body, mag_body）
# 6) うまく計算できなかったフレーム数などをQCとして集計する
# =============================================================

import cv2
import numpy as np
import pandas as pd
from collections import deque

# =============================================================
# 0) 入出力パス
# =============================================================
video_path = "/content/FBTCS_qt.mp4"      # 入力動画（切り出し済み）
npz_path   = "/content/track.npz"        # 以前のセルで作った骨格データ（NPZ）

out_video  = "/content/flow.mp4"         # 確認用: ROIとflowを重ねた動画
out_csv    = "/content/flow.csv"         # 解析用: 数値（vx, vy, mag, spike）
qc_csv     = "/content/flow_qc_summary.csv"  # QC集計

# =============================================================
# 1) パラメータ
# =============================================================
MAD_Z_THR   = 6.0     # 「動きが急に大きい」フレームをスパイクとみなす閾値
DRAW_SPIKE  = True    # スパイク時にROI枠の色を変える
WIN_SEC_MAD = 10.0    # スパイク判定に使う履歴の長さ（秒）

# Farnebäck法の設定（Optical Flowの計算方法のパラメータ）
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
# 2) 手動で決めた BED polygon = ROI
# ----------------------------------------------------------
# poly_rot_global は「前のセルでスライダー調整して作ったベッド領域」
# これが無いとROIを作れないのでここで止める
assert poly_rot_global is not None, "BED polygon not set"
ROI_POINTS = poly_rot_global.astype(np.float32)  # (N,2)

# =============================================================
# 3) Skeleton（NPZ）を読み込む
# =============================================================
# track.npz の中には, フレームごとの骨格(17点)や頭の点が入っている
dat = np.load(npz_path, allow_pickle=True)

kpts     = dat["kpts_raw"]    # shape: (T, 17, 3)  ... (x, y, confidence)
head_xy  = dat["head_xy"]     # shape: (T, 2)      ... (x, y)
time_all = dat["time_all"]    # shape: (T,)        ... 各フレームの時刻(秒)
fps_npz  = float(dat["fps"])  # NPZ側で保存されていたfps

# =============================================================
# 4) head_xy の「短い欠損」だけ補間する
# =============================================================
# YOLOの検出は時々失敗して head_xy が NaN になる
# ここでは「短い欠損だけ」線形補間で埋める
# （長い欠損を埋めると嘘の動きが混ざりやすい）
head_xy_interp = head_xy.copy()

MAX_GAP_SEC = 0.8
MAX_GAP_FRAMES = int(round(MAX_GAP_SEC * fps_npz))

for d in range(2):  # d=0: x, d=1: y
    s = pd.Series(head_xy_interp[:, d])

    # limit=MAX_GAP_FRAMES:
    #   連続欠損がMAX_GAP_FRAMES以下のときだけ補間する
    s_interp = s.interpolate(
        method="linear",
        limit=MAX_GAP_FRAMES,
        limit_direction="both"
    )

    head_xy_interp[:, d] = s_interp.values

# 補間の前後で, NaNフレーム数がどれくらい減ったか確認
n_before = np.sum(~np.all(np.isfinite(head_xy), axis=1))
n_after  = np.sum(~np.all(np.isfinite(head_xy_interp), axis=1))

print("\n=== head_xy interpolation QC ===")
print(f"NaN frames before : {n_before}")
print(f"NaN frames after  : {n_after}")
print(f"Max gap allowed   : {MAX_GAP_FRAMES} frames (~{MAX_GAP_SEC:.2f} sec)")

# =============================================================
# 5) COCO骨格で肩のindexを決める
# =============================================================
# COCO形式では:
#   5 = left shoulder
#   6 = right shoulder
LS = 5
RS = 6

# =============================================================
# 6) 動画を開く（Optical Flowは連続フレームが必要）
# =============================================================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"VideoCapture failed: {video_path}")

# 動画側fps（取れない場合もあるので保険）
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = fps_npz if fps_npz > 0 else 30.0

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 確認用動画を書き出すための writer
out = cv2.VideoWriter(
    out_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H)
)

# =============================================================
# 7) ROIマスクを作る
# =============================================================
# ROIの内側だけ平均を取りたいので, マスク画像を作っておく
mask = np.zeros((H, W), np.uint8)  # 0で埋めた画像
cv2.fillPoly(mask, [ROI_POINTS.astype(np.int32)], 1)  # ROI内を1にする
mask_bool = mask.astype(bool)  # True/Falseのマスクに変換

# =============================================================
# 8) 「動画の時刻 t_sec」から「NPZの骨格 index」を探す関数
# =============================================================
def skeleton_index_from_t(t_sec, time_all):
    # searchsorted:
    #   time_all の中で t_sec を入れる位置を探す
    ai = int(np.searchsorted(time_all, t_sec, side="right") - 1)
    # 範囲外にならないように0〜最後に丸める
    return int(np.clip(ai, 0, len(time_all) - 1))

# =============================================================
# 9) 身体座標系（ex, ey）を作る関数
# =============================================================
# 目的:
# ・Optical Flow は画像座標 (x右, y下) で出てくる
# ・でも「身体に沿った左右」「頭方向」成分が欲しい
#
# ここでは:
# ・ex = 左肩→右肩（左右方向ベクトル）
# ・ey = 肩の中心→頭（頭方向ベクトル）
# を作り, 2つが直交するように補正する（Gram-Schmidt）
def body_axes_from_kp(kp, head_xy_frame):
    LS_xy = kp[LS, :2]
    RS_xy = kp[RS, :2]

    # 肩や頭の座標が NaN なら軸を作れない
    if not (np.all(np.isfinite(LS_xy)) and
            np.all(np.isfinite(RS_xy)) and
            np.all(np.isfinite(head_xy_frame))):
        return None, None, False, "nan"

    shoulder_mid = (LS_xy + RS_xy) / 2.0

    # 左右方向
    ex = RS_xy - LS_xy
    # 頭方向
    ey = head_xy_frame - shoulder_mid

    # ベクトル長がほぼ0なら無理
    if np.linalg.norm(ex) < 1e-6 or np.linalg.norm(ey) < 1e-6:
        return None, None, False, "short"

    # 単位ベクトルにする（長さ1）
    ex = ex / np.linalg.norm(ex)

    # Gram–Schmidt: eyからex成分を引いて直交させる
    ey = ey - ex * np.dot(ey, ex)

    if np.linalg.norm(ey) < 1e-6:
        return None, None, False, "orth_fail"

    ey = ey / np.linalg.norm(ey)

    return ex.astype(np.float32), ey.astype(np.float32), True, "ok"

# =============================================================
# 10) MAD（スパイク判定）用の履歴バッファ
# =============================================================
# 直近 WIN_SEC_MAD 秒の mag_mean をためておく
mad_buf = deque(maxlen=max(20, int(round(WIN_SEC_MAD * fps))))

# =============================================================
# 11) QC用カウンタ（どこで失敗したか数える）
# =============================================================
cnt_total = 0
cnt_flow_none = 0
cnt_axes_invalid = 0
cnt_axes_nan = 0
cnt_axes_short = 0

# =============================================================
# 12) メインループ（フレームを順番に処理）
# =============================================================
flow_prev = None   # 1つ前のフレーム（グレー画像）を保存する
frame_idx = 0
rows = []          # CSV用に結果を溜める

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optical Flowはグレースケールで計算するのが一般的
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 現在時刻（秒）を取得
    # 取れない場合は frame_idx/fps で代用
    t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    t_sec = t_msec / 1000.0 if t_msec and t_msec > 0 else frame_idx / fps

    # 今の時刻に対応する骨格フレーム index を取得
    ai = skeleton_index_from_t(t_sec, time_all)
    kp = kpts[ai]
    head_pt = head_xy_interp[ai]

    cnt_total += 1

    # 身体軸を作る（作れない時もある）
    ex, ey, axes_valid, reason = body_axes_from_kp(kp, head_pt)

    # まずは NaN で初期化（失敗フレームは NaN のままCSVへ）
    vx_mean = np.nan
    vy_mean = np.nan
    mag_mean = np.nan

    overlay = frame.copy()  # 確認用の描画をする画像

    # 1枚目のフレームは「前フレーム」が無いので flow 計算不可
    if flow_prev is None:
        cnt_flow_none += 1

    # 軸が作れた場合だけ flow 計算する
    elif axes_valid:
        # ---------------------------------------------
        # 12-1) Optical Flow を計算
        # ---------------------------------------------
        flow = cv2.calcOpticalFlowFarneback(flow_prev, gray, None, **FB_PARAMS)
        fx, fy = flow[..., 0], flow[..., 1]  # 画像座標での動き（x成分, y成分）

        # ---------------------------------------------
        # 12-2) 画像座標の動きを「身体座標」へ変換
        # ---------------------------------------------
        # ex, ey は単位ベクトル（長さ1）なので, 内積で射影できる
        fx_body = fx * ex[0] + fy * ex[1]  # 左右方向成分
        fy_body = fx * ey[0] + fy * ey[1]  # 頭方向成分

        # 動きの大きさ（マグニチュード）
        mag_body = cv2.magnitude(fx_body, fy_body)

        # ---------------------------------------------
        # 12-3) ROI内だけ平均を取る
        # ---------------------------------------------
        vx_mean  = float(np.nanmean(fx_body[mask_bool]))
        vy_mean  = float(np.nanmean(fy_body[mask_bool]))
        mag_mean = float(np.nanmean(mag_body[mask_bool]))

        # ---------------------------------------------
        # 12-4) 確認用にflowを色で可視化（ROI内だけ重ねる）
        # ---------------------------------------------
        ang = np.arctan2(fy_body, fx_body)  # 動きの向き（角度）
        hsv = np.zeros((H, W, 3), np.uint8)
        hsv[..., 1] = 255
        hsv[..., 0] = ((ang + np.pi) / (2*np.pi) * 180).astype(np.uint8)
        hsv[..., 2] = cv2.normalize(mag_body, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # ROI内だけflowを重ねる
        overlay = cv2.addWeighted(
            frame, 0.35,
            flow_color * (mask_bool[..., None]),
            0.65,
            0
        )

    else:
        # 軸が作れなかった場合（肩/頭がNaNなど）
        cnt_axes_invalid += 1
        if reason == "nan":
            cnt_axes_nan += 1
        else:
            cnt_axes_short += 1

    # 次のフレーム用に保存（これが「前フレーム」になる）
    flow_prev = gray

    # ---------------------------------------------
    # 12-5) スパイク判定（MAD）
    # ---------------------------------------------
    # 注意:
    # ・スパイクを検出するだけで, データ除外はしていない
    spike_flag = 0

    if np.isfinite(mag_mean):
        mad_buf.append(mag_mean)

    if len(mad_buf) >= 20 and np.isfinite(mag_mean):
        arr = np.asarray(mad_buf)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) + 1e-9
        z = abs(mag_mean - med) / mad
        if z > MAD_Z_THR:
            spike_flag = 1

    # ---------------------------------------------
    # 12-6) ROI枠を描く（スパイクなら赤）
    # ---------------------------------------------
    color = (0, 0, 255) if (spike_flag and DRAW_SPIKE) else (0, 255, 255)
    cv2.polylines(overlay, [ROI_POINTS.astype(int)], True, color, 2)

    # 確認用動画へ保存
    out.write(overlay)

    # CSV用に保存
    rows.append([frame_idx, t_sec, vx_mean, vy_mean, mag_mean, spike_flag])

    frame_idx += 1

# 動画を閉じる
cap.release()
out.release()

# =============================================================
# 13) CSV 保存（解析で使うメイン出力）
# =============================================================
df_out = pd.DataFrame(
    rows,
    columns=["frame", "t_sec", "vx_body", "vy_body", "mag_body", "spike"]
)
df_out.to_csv(out_csv, index=False)

# =============================================================
# 14) QC Summary（どれだけ有効データが取れたか）
# =============================================================
mag = df_out["mag_body"].to_numpy()
valid = np.isfinite(mag)  # mag_bodyがNaNでないフレームだけ True

T_total = len(valid)
T_valid = int(valid.sum())
T_excl  = T_total - T_valid
valid_rate = T_valid / T_total if T_total > 0 else np.nan

# 「連続して何フレーム NaN だったか」の最大値を調べる
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
print(f"flow_prev None         : {cnt_flow_none}   (最初の1フレームは必ずここ)")
print(f"axes invalid (total)   : {cnt_axes_invalid}")
print(f"  - NaN keypoints      : {cnt_axes_nan}")
print(f"  - short/orth failure : {cnt_axes_short}")

# =============================================================
# 15) QC CSV（症例ごとのQCを表で集計しやすくする）
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
