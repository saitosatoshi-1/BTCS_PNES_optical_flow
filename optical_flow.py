"""
body_axis_flow_beginner.py

目的.
  連続フレームの画像から optical flow を計算し,
  upstreamで推定した body axis (ex, ey) に沿った動き(vx_body, vy_body)を,
  ROI(多角形)の中で平均してCSVに保存する.

重要ポイント(初心者向け).
  - optical flowは「前フレーム」が必要なので, 最初の1フレームはNaNになる.
  - ex/ey がNaNのフレームは, upstreamの推定が失敗しているので, 出力もNaNにする.
  - 動画とupstreamの時間ズレに対応するため, 現在フレーム時刻(t_sec)から
    upstreamの配列index(skel_idx)を引く.

入出力.
  input:
    - video_path: 入力動画
    - inter_npz: 事前に作ったNPZ (time_all, fps, ex, ey を含む)
    - roi_polygon_xy: ROI多角形 (N,2) 画像座標
  output:
    - out_csv: frameごとの特徴量
    - out_qc_mp4: ROI内だけflow可視化を重ねたQC動画(任意)
    - out_qc_csv: カウントなどのQCサマリ
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import cv2


# ==========================================================
# 1) Farnebäck optical flow パラメータ
#    まずはこのままでOK. 迷ったらwinsizeだけ調整することが多い.
# ==========================================================
FB_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)


# ==========================================================
# 2) 小さな便利関数
# ==========================================================
def open_video(video_path: str, fallback_fps: float) -> tuple[cv2.VideoCapture, float, int, int]:
    """
    動画を開いて, fps, 幅W, 高さHを返す.

    fallback_fps:
      動画からfpsが取れない時に使う. upstreamのfpsを渡すのが安全.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"VideoCapture failed: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = float(fallback_fps)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, float(fps), W, H


def make_writer(out_mp4: str, fps: float, W: int, H: int) -> cv2.VideoWriter:
    """
    QC動画を書き出すVideoWriterを作る.
    """
    return cv2.VideoWriter(
        out_mp4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (W, H)
    )


def build_roi_mask(H: int, W: int, roi_polygon_xy: np.ndarray) -> np.ndarray:
    """
    ROI多角形から, 画像サイズ(H,W)の boolean mask を作る.

    True: ROIの内側
    False: ROIの外側
    """
    roi_polygon_xy = np.asarray(roi_polygon_xy, dtype=np.int32)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon_xy], 1)
    return mask.astype(bool)


def frame_time_sec(cap: cv2.VideoCapture, frame_idx: int, fps: float) -> float:
    """
    現在フレームの時刻(秒)を返す.

    できればCAP_PROP_POS_MSECを使う.
    取れない時は frame_idx / fps にフォールバック.
    """
    t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if t_msec is not None and t_msec > 0:
        return float(t_msec) / 1000.0
    return float(frame_idx) / float(fps)


def skel_index_from_time(t_sec: float, time_all: np.ndarray) -> int:
    """
    動画の時刻t_secに対応するupstream index を返す.

    ルール(重要).
      - time_all[idx] <= t_sec を満たす最大のidxを選ぶ(未来を見ない).
      - 範囲外は端にクリップする.
    """
    idx = int(np.searchsorted(time_all, t_sec, side="right") - 1)
    idx = int(np.clip(idx, 0, len(time_all) - 1))
    return idx


def darken(frame_bgr: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    QC用. 計算できないフレームは暗くして分かりやすくする.

    alpha:
      0.0 -> 変化なし
      0.6 -> よく使う
    """
    black = np.zeros_like(frame_bgr)
    return cv2.addWeighted(frame_bgr, 1 - alpha, black, alpha, 0)


def flow_to_color(fx_body: np.ndarray, fy_body: np.ndarray) -> np.ndarray:
    """
    body座標のflowから, 色画像(flow_color)を作る.

    表示ルール.
      - Hue: 方向
      - Value: 大きさ(明るさ)
    """
    mag = cv2.magnitude(fx_body, fy_body)
    ang = np.arctan2(fy_body, fx_body)

    hsv = np.zeros((fx_body.shape[0], fx_body.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    # OpenCVのHueは0..180
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)

    # magを0..255へ正規化して明るさにする
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_color


def overlay_roi_flow(
    frame_bgr: np.ndarray,
    flow_color_bgr: np.ndarray,
    roi_mask: np.ndarray,
    roi_polygon_xy: np.ndarray
) -> np.ndarray:
    """
    ROIの内側だけ, flowの可視化を半透明で重ねる.
    """
    blended = cv2.addWeighted(frame_bgr, 0.5, flow_color_bgr, 0.5, 0)
    out = frame_bgr.copy()
    out[roi_mask] = blended[roi_mask]

    # ROI境界を描いておく
    cv2.polylines(out, [np.asarray(roi_polygon_xy, dtype=np.int32)], True, (0, 255, 255), 2)
    return out


def compute_body_axis_features(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    roi_mask: np.ndarray,
    fb_params: dict
) -> tuple[float, float, float, np.ndarray]:
    """
    1ステップ分の特徴量を計算する.

    手順.
      1) dense flow (fx, fy) を計算
      2) (fx, fy) を ex と ey に射影して (fx_body, fy_body) を作る
      3) ROI内平均で vx_mean, vy_mean, mag_mean を作る
      4) QC用に flow_color を作る
    """
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **fb_params)
    fx = flow[..., 0]
    fy = flow[..., 1]

    # 射影: dot((fx,fy), ex) と dot((fx,fy), ey)
    fx_body = fx * ex[0] + fy * ex[1]
    fy_body = fx * ey[0] + fy * ey[1]

    mag_body = cv2.magnitude(fx_body, fy_body)

    vx_mean = float(np.nanmean(fx_body[roi_mask]))
    vy_mean = float(np.nanmean(fy_body[roi_mask]))
    mag_mean = float(np.nanmean(mag_body[roi_mask]))

    flow_color = flow_to_color(fx_body, fy_body)
    return vx_mean, vy_mean, mag_mean, flow_color


# ==========================================================
# 3) メイン処理
# ==========================================================
def run_body_axis_flow(
    video_path: str,
    inter_npz: str,
    roi_polygon_xy: np.ndarray,
    out_csv: str,
    out_qc_csv: str,
    out_qc_mp4: str | None = None,
) -> None:
    """
    メイン関数.
    """
    # --- upstream NPZを読む ---
    dat = np.load(inter_npz, allow_pickle=True)

    time_all = dat["time_all"]            # (T,)
    fps_npz = float(dat["fps"])           # upstreamのfps
    ex_all = dat["ex"]                    # (T,2)
    ey_all = dat["ey"]                    # (T,2)

    # metaは無くても動くようにする(初心者向けに頑丈に)
    meta = None
    if "meta" in dat.files:
        try:
            meta = dat["meta"][0]
        except Exception:
            meta = None

    # --- 動画を開く ---
    cap, fps_vid, W, H = open_video(video_path, fallback_fps=fps_npz)

    # --- ROI mask作成 ---
    roi_mask = build_roi_mask(H, W, roi_polygon_xy)

    # --- QC動画writer(任意) ---
    writer = None
    if out_qc_mp4 is not None:
        writer = make_writer(out_qc_mp4, fps_vid, W, H)

    # --- ループ用の変数 ---
    rows = []
    prev_gray = None
    frame_idx = 0

    # QC counters
    cnt_first_frame = 0
    cnt_axes_invalid = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 現在フレームの時刻(秒)
        t_sec = frame_time_sec(cap, frame_idx, fps_vid)

        # upstream配列index
        skel_idx = skel_index_from_time(t_sec, time_all)

        # body axisを取り出す
        ex = ex_all[skel_idx]
        ey = ey_all[skel_idx]

        axes_ok = bool(np.isfinite(ex).all() and np.isfinite(ey).all())

        # デフォルト: NaN
        vx = np.nan
        vy = np.nan
        mag = np.nan

        # QC用フレーム
        qc_frame = darken(frame, alpha=0.6)

        if not axes_ok:
            # upstreamの推定失敗
            cnt_axes_invalid += 1
            cv2.polylines(qc_frame, [np.asarray(roi_polygon_xy, dtype=np.int32)], True, (0, 255, 255), 2)

        elif prev_gray is None:
            # 最初のフレームはflowが作れない
            cnt_first_frame += 1
            cv2.polylines(qc_frame, [np.asarray(roi_polygon_xy, dtype=np.int32)], True, (0, 255, 255), 2)

        else:
            # 通常計算
            vx, vy, mag, flow_color = compute_body_axis_features(
                prev_gray, gray, ex, ey, roi_mask, FB_PARAMS
            )
            qc_frame = overlay_roi_flow(frame, flow_color, roi_mask, roi_polygon_xy)

        # 結果保存(1フレーム1行)
        rows.append([frame_idx, t_sec, skel_idx, int(axes_ok), vx, vy, mag])

        if writer is not None:
            writer.write(qc_frame)

        prev_gray = gray
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    # --- CSV出力 ---
    df = pd.DataFrame(
        rows,
        columns=["frame", "t_sec", "skel_idx", "axes_ok", "vx_body", "vy_body", "mag_body"]
    )
    df.to_csv(out_csv, index=False)

    # --- QC summary出力 ---
    qc = {
        "video_path": video_path,
        "inter_npz": inter_npz,
        "fps_video": float(fps_vid),
        "fps_npz": float(fps_npz),
        "n_frames": int(frame_idx),
        "cnt_first_frame_no_flow": int(cnt_first_frame),
        "cnt_axes_invalid": int(cnt_axes_invalid),
        "meta": meta,
    }
    pd.DataFrame([qc]).to_csv(out_qc_csv, index=False)


# ==========================================================
# 4) 使い方例
# ==========================================================
if __name__ == "__main__":
    # 入力
    video_path = "input.mp4"
    inter_npz = "skeleton_pc1.npz"

    # ROI多角形(ダミー例). 実際は自分の動画に合わせて座標を入れる.
    roi_polygon_xy = np.array([
        [100, 100],
        [500, 120],
        [520, 380],
        [120, 400],
    ], dtype=float)

    # 出力
    out_csv = "flow.csv"
    out_qc_csv = "flow_qc_summary.csv"
    out_qc_mp4 = "flow_qc.mp4"

    run_body_axis_flow(
        video_path=video_path,
        inter_npz=inter_npz,
        roi_polygon_xy=roi_polygon_xy,
        out_csv=out_csv,
        out_qc_csv=out_qc_csv,
        out_qc_mp4=out_qc_mp4,
    )

    print("Done.")
