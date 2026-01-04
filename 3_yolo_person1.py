# ==========================================================
# Person1 Tracking + Pose Estimation + RAW NPZ Export
#
# 【このセルの役割】
#   ・動画内に複数人（患者＋介助者など）が映っていても,
#     「ベッド上の人物（person1）」だけを安定して追跡する
#   ・YOLO Pose を用いて 17点骨格（COCO形式）を推定する
#   ・後段解析（PCA, ADS, 周期解析など）のために,
#     前処理をほぼ行わない「RAW データ」を NPZ として保存する
#
# 【設計思想】
#   ・IoUトラッキングは使わない
#     → 介助者に飛びやすいため
#   ・「ベッド内にいるか」「前フレームとの連続性」だけを使う
#   ・失敗しても止まらず, 欠損は NaN で保存する
# ==========================================================

import os
import numpy as np
import cv2
from ultralytics import YOLO

# ----------------------------------------------------------
# I/O パス設定
# ----------------------------------------------------------
# 入力動画（事前に trim・fps 正規化済みを想定）
video_path   = "/content/FBTCS_qt.mp4"

# person1 を追跡した様子を描画した動画（中間確認用）
out_try_path = "/content/track.mp4"

# QuickTime 互換（macOSで確実に再生できる形式）
out_qt_path  = "/content/track_qt.mp4"

# 追跡状態のログ（今回は主にデバッグ用）
log_path     = "/content/events.log"

# 骨格の RAW データ保存先
npz_out      = "/content/track.npz"

# ----------------------------------------------------------
# 手動で決めた BED polygon を読み込む
# ----------------------------------------------------------
# poly_rot_global は, 事前 UI で指定した「ベッド領域」
# （このセルより前で定義されている前提）
print(poly_rot_global)
assert poly_rot_global is not None, "BED polygon not set"

# polygon を OpenCV 用に整形
priority_coords = poly_rot_global.astype(np.float32)
gate_poly = priority_coords.reshape(-1,1,2).astype(np.float32)

# ベッド中心座標
# INIT 状態で「最初に選ぶ person1」の基準に使う
bed_cx = float(priority_coords[:,0].mean())
bed_cy = float(priority_coords[:,1].mean())

# ベッド外接矩形（※ 可視化用のみ）
bx1, by1 = priority_coords.min(axis=0)
bx2, by2 = priority_coords.max(axis=0)

# ----------------------------------------------------------
# YOLO Pose モデル
# ----------------------------------------------------------
# x モデルは重いが, 臨床動画では安定性を優先
pose_model = YOLO("yolo11x-pose.pt")

# ----------------------------------------------------------
# Utility 関数群
# ----------------------------------------------------------
def center_from_kpts(kp_xy):
    """
    有効な keypoint だけを用いて
    人物全体の代表中心座標を計算する

    kp_xy : (17,2) array
    戻り値:
      center (x,y) or None
    """
    valid = ~np.isnan(kp_xy).any(axis=1)
    if not np.any(valid):
        return None
    return np.nanmean(kp_xy[valid], axis=0)


def inside_gate(x, y):
    """
    点 (x, y) が BED polygon 内にあるかどうか
    True ならベッド上の人物とみなす
    """
    return cv2.pointPolygonTest(
        gate_poly,
        (float(x), float(y)),
        False
    ) >= 0


def compute_head_point(kp_xy, kp_c, conf_thr=0.3):
    """
    頭部代表点を計算する関数

    優先順位:
      1) nose（最も解剖学的に分かりやすい）
      2) 両眼・両耳のうち, 検出できた点の重心

    研究計画に記載したロジックと完全に一致させている
    """
    # --- nose ---
    if kp_c[0] >= conf_thr and np.isfinite(kp_xy[0]).all():
        return kp_xy[0].copy(), "nose"

    # --- eyes + ears ---
    face_idx = [1, 2, 3, 4]
    valid = [
        i for i in face_idx
        if kp_c[i] >= conf_thr and np.isfinite(kp_xy[i]).all()
    ]

    if len(valid) > 0:
        return np.mean(kp_xy[valid], axis=0), "face"

    # 頭部が全く取れない場合
    return np.array([np.nan, np.nan]), "none"


# YOLO Pose の骨格接続定義（可視化用）
SKELETON = [
    (0,1),(1,3),(0,2),(2,4),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]


def draw_skeleton(img, kp_xy, color=(0,0,255), th=2):
    """
    骨格を画像に描画する（解析ではなく確認用）
    """
    for a, b in SKELETON:
        xa, ya = kp_xy[a]
        xb, yb = kp_xy[b]
        if (np.isfinite(xa) and np.isfinite(ya)
            and np.isfinite(xb) and np.isfinite(yb)):
            cv2.line(
                img,
                (int(xa), int(ya)),
                (int(xb), int(yb)),
                color,
                th
            )

# ----------------------------------------------------------
# 動画オープン
# ----------------------------------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"VideoCapture failed: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    out_try_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H)
)

# ----------------------------------------------------------
# 状態管理変数
# ----------------------------------------------------------
# state:
#   INIT    : まだ person1 が確定していない
#   LOCKED  : person1 を追跡中
#   MISSING : ベッド内に人物が検出できない
state = "INIT"

person1_idx = None
prev_center = None

# フレームごとに保存する配列
all_kpts        = []   # (17,3) keypoints
all_state       = []   # 状態文字列
all_idx         = []   # person index
all_head_xy     = []   # 頭部代表点 (x,y)
all_head_source = []   # nose / face / none / missing

# ----------------------------------------------------------
# Tracking / Pose 用パラメータ
# ----------------------------------------------------------
CONF_MIN    = 0.3   # keypoint 信頼度の下限
MAX_JUMP_PX = 120   # フレーム間の最大許容移動量（今回は未使用）

# ----------------------------------------------------------
# メインループ（1フレームずつ処理）
# ----------------------------------------------------------
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Pose 推定
    res = pose_model.predict(frame, verbose=False, conf=CONF_MIN)[0]

    boxes = (
        res.boxes.xyxy.cpu().numpy()
        if res.boxes is not None else np.zeros((0,4))
    )
    kpts = (
        res.keypoints.data.cpu().numpy()
        if res.keypoints is not None else np.zeros((0,17,3))
    )

    # -----------------------------
    # 人物が検出されなかった場合
    # -----------------------------
    if kpts.shape[0] == 0:
        all_kpts.append(
            all_kpts[-1] if frame_idx > 0 else np.full((17,3), np.nan)
        )
        all_state.append("MISSING")
        all_idx.append(-1)
        all_head_xy.append(np.array([np.nan, np.nan]))
        all_head_source.append("missing")
        frame_idx += 1
        continue

    # -----------------------------
    # ベッド内にいる人物を候補に
    # -----------------------------
    candidates = []
    for i in range(kpts.shape[0]):
        kp_xy = kpts[i,:,:2].copy()
        kp_c  = kpts[i,:,2]

        # 低信頼度 keypoint は NaN に
        kp_xy[kp_c < CONF_MIN] = np.nan

        center = center_from_kpts(kp_xy)
        if center is None:
            x1,y1,x2,y2 = boxes[i]
            center = np.array([(x1+x2)/2, (y1+y2)/2])

        if inside_gate(center[0], center[1]):
            candidates.append(
                dict(i=i, center=center, kp_xy=kp_xy, kp_c=kp_c)
            )

    # 候補がいない場合
    if not candidates:
        all_kpts.append(all_kpts[-1])
        all_state.append("MISSING")
        all_idx.append(-1)
        all_head_xy.append(np.array([np.nan, np.nan]))
        all_head_source.append("missing")
        frame_idx += 1
        continue

    # INIT / MISSING 状態ではベッド中心に一番近い人物を選ぶ
    chosen = min(
        candidates,
        key=lambda c: np.hypot(
            c["center"][0] - bed_cx,
            c["center"][1] - bed_cy
        )
    )

    state = "LOCKED"
    person1_idx = chosen["i"]
    prev_center = chosen["center"]

    # keypoints を (17,3) で保存
    row_k = np.stack(
        [chosen["kp_xy"][:,0], chosen["kp_xy"][:,1], chosen["kp_c"]],
        axis=1
    )

    # 頭部代表点を計算
    head_xy, head_src = compute_head_point(
        chosen["kp_xy"],
        chosen["kp_c"],
        CONF_MIN
    )

    # 保存
    all_kpts.append(row_k)
    all_state.append(state)
    all_idx.append(person1_idx)
    all_head_xy.append(head_xy)
    all_head_source.append(head_src)

    # 可視化（確認用）
    draw_skeleton(frame, chosen["kp_xy"])
    out.write(frame)

    frame_idx += 1

# ----------------------------------------------------------
# 後処理
# ----------------------------------------------------------
cap.release()
out.release()

kpts_raw  = np.array(all_kpts)
state_arr = np.array(all_state)
time_all  = np.arange(len(kpts_raw)) / fps

# RAW NPZ 保存
np.savez(
    npz_out,
    kpts_raw=kpts_raw,
    head_xy=np.array(all_head_xy),
    head_source=np.array(all_head_source),
    fps=float(fps),
    time_all=time_all,
    state=state_arr,
    person1_idx=np.array(all_idx),
)

print("[OK] NPZ saved:", npz_out)

# ----------------------------------------------------------
# QuickTime 互換 mp4 に再エンコード
# ----------------------------------------------------------
os.system(
    f"ffmpeg -y -i {out_try_path} "
    f"-vcodec libx264 -pix_fmt yuv420p "
    f"-profile:v baseline -level 3.0 "
    f"-movflags +faststart {out_qt_path}"
)

print("[OK] QT video:", out_qt_path)
