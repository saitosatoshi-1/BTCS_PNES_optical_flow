import numpy as np
import cv2
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider

# ==========================================================
# このセルの目的（初心者向け）
# ----------------------------------------------------------
# 1) 動画の「ある1フレーム」を1枚だけ取り出して表示する.
# 2) その画像の上に「ベッドの枠（四角形）」を線で重ねる.
# 3) スライダーで四角形の4頂点の位置を動かしたり,
#    全体を回転させたりして, ベッドの位置を手作業で合わせる.
#
# よくある用途:
# ・VEEG動画でベッド領域を正確に決めたい.
# ・その座標を gate_config.json に保存したい.
# ==========================================================


# ==========================================================
# ① まず, 表示したいフレームを1枚読み込む
# ==========================================================

# best_frame_idx:
#   何フレーム目を表示するか（0,1,2,...）
#   例: 30 なら「30番目のフレーム」を表示する
best_frame_idx = 30

# 対象動画ファイル（事前に切り出した動画を想定）
video_path = "/content/FBTCS_qt.mp4"

# OpenCVで動画を開く
cap = cv2.VideoCapture(video_path)

# 開けなかったらエラー（ファイルパスが違う, コーデック未対応など）
if not cap.isOpened():
    raise RuntimeError(f"VideoCapture failed: {video_path}")

# 動画の「読み出し位置」を best_frame_idx フレームに移動する
# ※動画は「秒」ではなく「フレーム番号」で移動する.
cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)

# 念のため1フレーム捨て読み（環境によってはシーク直後が不安定なことがあるため）
ret, _ = cap.read()

# もう一度, 取りたいフレームに戻す
cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)

# 目的のフレームを読み込む
ret, frame = cap.read()

# 動画を閉じる（資源の解放. 忘れると不具合の原因になる）
cap.release()

# フレーム取得に失敗したら停止（範囲外, 読めないフレームなど）
if not ret:
    raise RuntimeError("Failed to load frame")

# frame は画像（numpy配列）で, shape は (高さ, 幅, 色の3要素)
# H: 画像の高さ, W: 画像の幅
H, W = frame.shape[:2]


# ==========================================================
# ② ベッド枠（四角形ポリゴン）の初期値を作る
# ==========================================================
# いきなり完全に合う四角形を作るのは難しいので,
# まず「だいたいベッドっぽい位置」に四角形を置く.
#
# W*0.25 のように「画面サイズに対する割合」で決めると,
# 動画サイズが変わっても初期位置がそれなりになるメリットがある.

# tL = top-left  （左上）
# tR = top-right （右上）
# bR = bottom-right（右下）
# bL = bottom-left （左下）
tL = np.array([W * 0.25, H * 0.30])
tR = np.array([W * 0.75, H * 0.30])
bR = np.array([W * 0.75, H * 0.70])
bL = np.array([W * 0.25, H * 0.70])

# 回転後のポリゴンを外で参照したい時のために入れている変数
# （このセル内では主に表示用. 他セルで拾ってjson保存する等に使える）
poly_rot_global = None


# ==========================================================
# ③ スライダーを動かした時に毎回呼ばれる関数
# ==========================================================
def update(
    tL_x, tL_y,
    tR_x, tR_y,
    bR_x, bR_y,
    bL_x, bL_y,
    rotate_deg
):
    """
    この関数は, スライダーが動くたびに自動で呼ばれる.

    入ってくる値:
      - tL_x, tL_y ... 左上のx,y
      - tR_x, tR_y ... 右上のx,y
      - bR_x, bR_y ... 右下のx,y
      - bL_x, bL_y ... 左下のx,y
      - rotate_deg   ... 全体を回転させる角度（度）

    やること:
      1) スライダー値から四角形を作る
      2) 四角形を中心で回転させる
      3) 画像に赤線で重ねて表示する
      4) 今の座標（json保存用）をprintする
    """
    global poly_rot_global

    # ------------------------------------------------------
    # ③-1) スライダー値からポリゴン（四角形）を作る
    # ------------------------------------------------------
    # poly は [ [x,y], [x,y], [x,y], [x,y] ] の形
    # 頂点の順番は非常に大事:
    #   [左上, 右上, 右下, 左下]
    poly = np.array(
        [
            [tL_x, tL_y],
            [tR_x, tR_y],
            [bR_x, bR_y],
            [bL_x, bL_y],
        ],
        dtype=float
    )

    # ------------------------------------------------------
    # ③-2) 回転処理（ポリゴン中心を基準に回す）
    # ------------------------------------------------------
    # rotate_deg は「度」なので, 数学計算で使う「ラジアン」に変換する
    theta = np.deg2rad(rotate_deg)

    # 2次元の回転行列 R
    # これを使うと点 (x,y) を回転できる
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ],
        dtype=float
    )

    # 四角形の中心（4点の平均）を回転中心にする
    center = poly.mean(axis=0)

    # 回転の基本手順:
    # 1) center を引いて「中心が原点(0,0)」になるように移動
    # 2) 回転行列で回す
    # 3) center を足して元の位置に戻す
    poly_rot = (poly - center) @ R.T + center

    # 他セルで使えるようにコピーして保持
    poly_rot_global = poly_rot.copy()

    # ------------------------------------------------------
    # ③-3) 画像に描画して表示する
    # ------------------------------------------------------
    # 元画像を直接いじると戻せないので, copy() してから線を描く
    img = frame.copy()

    # polylines は「線で囲む」関数
    # OpenCVは整数座標が必要なので astype(int) にしている
    cv2.polylines(
        img,
        [poly_rot.astype(int)],
        isClosed=True,
        color=(0, 0, 255),   # 赤（OpenCVはBGR順）
        thickness=3
    )

    # matplotlib で表示する
    # 注意: OpenCVの色はBGR, matplotlibはRGBなので変換が必要
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Adjust BED polygon (with rotation)")
    plt.axis("off")
    plt.show()

    # ------------------------------------------------------
    # ③-4) 現在の座標を表示（json保存用）
    # ------------------------------------------------------
    # ここに出る座標をそのまま gate_config.json に貼る想定
    print("\n=== CURRENT ROTATED POLYGON COORDS ===")
    print(poly_rot)


# ==========================================================
# ④ スライダー（UI）を作る
# ==========================================================
# IntSlider: 整数のスライダー（x,y座標は整数で十分なので）
# FloatSlider: 小数も動かせるスライダー（回転角度に使う）
#
# min/max は画面の端:
#   x は 0〜W
#   y は 0〜H
#
# step=2 は「2ピクセルずつ」動く. 細かくしたければ 1 にする.
interact(
    update,
    tL_x = IntSlider(min=0, max=W, value=int(tL[0]), step=2, description="tL_x"),
    tL_y = IntSlider(min=0, max=H, value=int(tL[1]), step=2, description="tL_y"),

    tR_x = IntSlider(min=0, max=W, value=int(tR[0]), step=2, description="tR_x"),
    tR_y = IntSlider(min=0, max=H, value=int(tR[1]), step=2, description="tR_y"),

    bR_x = IntSlider(min=0, max=W, value=int(bR[0]), step=2, description="bR_x"),
    bR_y = IntSlider(min=0, max=H, value=int(bR[1]), step=2, description="bR_y"),

    bL_x = IntSlider(min=0, max=W, value=int(bL[0]), step=2, description="bL_x"),
    bL_y = IntSlider(min=0, max=H, value=int(bL[1]), step=2, description="bL_y"),

    rotate_deg = FloatSlider(min=-45, max=45, step=1, value=0, description="Rotate")
)
