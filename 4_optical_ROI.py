import numpy as np
import cv2
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider

# ==========================================================
# 目的:
#   動画の特定フレームを1枚表示し,
#   その上に「ベッド領域（四角形ポリゴン）」を重ねて
#   スライダー操作で位置・形・回転を微調整する.
#
# 想定用途:
#   ・VEEG動画でベッド位置を手動で正確に決めたい
#   ・gate_config.json 用の polygon 座標を作る
# ==========================================================


# ----------------------------------------------------------
# 表示したいフレームを読み込む
# ----------------------------------------------------------
# best_frame_idx:
#   ベッドや患者がはっきり写っているフレーム番号
best_frame_idx = 30

# 対象動画（事前に trim 済み mp4 を想定）
video_path = "/content/FBTCS_qt.mp4"

# OpenCVで動画を開き, 指定フレームに移動
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"VideoCapture failed: {video_path}")
cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)

# 念のため1フレーム捨て読み
ret, _ = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
ret, frame = cap.read()
cap.release()

# フレーム取得に失敗した場合は停止
if not ret:
    raise RuntimeError("Failed to load frame")

# フレームサイズ（高さ H, 幅 W）
H, W = frame.shape[:2]


# ----------------------------------------------------------
# 初期ポリゴン（だいたいの四角形）
# ----------------------------------------------------------
# 画面サイズに対する比率で初期値を置くことで,
# 動画サイズが変わってもそれなりの位置から開始できる
tL = np.array([W*0.25, H*0.30])  # 左上 (top-left)
tR = np.array([W*0.75, H*0.30])  # 右上 (top-right)
bR = np.array([W*0.75, H*0.70])  # 右下 (bottom-right)
bL = np.array([W*0.25, H*0.70])  # 左下 (bottom-left)

poly_rot_global = None
# ----------------------------------------------------------
# スライダーが動くたびに呼ばれる更新関数
# ----------------------------------------------------------
def update(
    tL_x, tL_y,
    tR_x, tR_y,
    bR_x, bR_y,
    bL_x, bL_y,
    rotate_deg
):
    """
    スライダー値を受け取り,
    ・ポリゴンを作成
    ・中心回転を適用
    ・画像に描画して表示
    """
    global poly_rot_global
    # ---- ポリゴンを座標配列として構築 ----
    # 頂点の順序は [左上, 右上, 右下, 左下]
    poly = np.array([
        [tL_x, tL_y],
        [tR_x, tR_y],
        [bR_x, bR_y],
        [bL_x, bL_y],
    ], float)

    # ------------------------------------------------------
    # 回転処理（ポリゴン中心を基準）
    # ------------------------------------------------------
    # rotate_deg: 度 → ラジアンに変換
    theta = np.deg2rad(rotate_deg)

    # 2次元回転行列
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # ポリゴンの重心（回転の中心）
    center = poly.mean(axis=0)

    # 中心を原点に移動 → 回転 → 元の位置に戻す
    poly_rot = (poly - center) @ R.T + center
    poly_rot_global = poly_rot.copy()
    # ------------------------------------------------------
    # 描画
    # ------------------------------------------------------
    img = frame.copy()

    # 回転後ポリゴンを赤線で描画
    cv2.polylines(
        img,
        [poly_rot.astype(int)],
        isClosed=True,
        color=(0, 0, 255),  # 赤 (BGR)
        thickness=3
    )

    # matplotlib で表示（OpenCVはBGRなのでRGBに変換）
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Adjust BED polygon (with rotation)")
    plt.axis("off")
    plt.show()

    # 現在のポリゴン座標を表示（json保存用）
    print("\n=== CURRENT ROTATED POLYGON COORDS ===")
    print(poly_rot)


# ----------------------------------------------------------
# スライダーUI
# ----------------------------------------------------------
# 各頂点の x, y を個別に調整できる
# rotate_deg で全体を回転できる
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

