# ==========================================================
# Person1 Tracking + Pose Estimation + RAW NPZ Export
#
# このセルでやっていること（超ざっくり）:
# 1) 動画を1フレームずつ読む
# 2) YOLO Pose で「人」と「骨格(17点)」を検出する
# 3) ベッドの中にいる人だけを候補にする
# 4) 候補の中から「患者っぽい人」を1人選ぶ（person1）
# 5) person1 の骨格を毎フレーム保存する（欠けた所は NaN）
# 6) 確認用に骨格を描いた動画も保存する
#
# なぜ NPZ で保存する?
# ・後から解析（PCA, ADS, 周期解析など）をするときに,
#   まず「生の骨格データ」が必要だから
# ・NPZ は numpy配列をそのまま保存できて速くて便利
# ==========================================================

import os
import numpy as np
import cv2
from ultralytics import YOLO

# ----------------------------------------------------------
# 0) ファイルパス（入力と出力）
# ----------------------------------------------------------
# 入力動画（切り出し済みの mp4 を想定）
video_path   = "/content/FBTCS_qt.mp4"

# 追跡の様子を描いた動画（確認用）
out_try_path = "/content/track.mp4"

# macOSのQuickTimeで確実に再生できる形式にした動画
out_qt_path  = "/content/track_qt.mp4"

# （今回はほぼ使っていない）ログファイル
log_path     = "/content/events.log"

# 骨格の「生データ」を保存する NPZ
npz_out      = "/content/track.npz"

# ----------------------------------------------------------
# 1) ベッド領域（BED polygon）の読み込み
# ----------------------------------------------------------
# 注意:
#   poly_rot_global は「前のセル（UI調整）」で作った変数
#   つまり, このセル単体では動かず,
#   先にベッド領域をスライダーで決めている必要がある
print(poly_rot_global)
assert poly_rot_global is not None, "BED polygon not set"

# OpenCV の pointPolygonTest が使える形に整形する
# (N,2) -> (N,1,2) という形にする
priority_coords = poly_rot_global.astype(np.float32)
gate_poly = priority_coords.reshape(-1, 1, 2).astype(np.float32)

# ベッド中心座標（候補から「ベッド中心に近い人」を選ぶため）
bed_cx = float(priority_coords[:, 0].mean())
bed_cy = float(priority_coords[:, 1].mean())

# ベッドの外接矩形（これは「見た目用」, 解析にはほぼ使わない）
bx1, by1 = priority_coords.min(axis=0)
bx2, by2 = priority_coords.max(axis=0)

# ----------------------------------------------------------
# 2) YOLO Pose モデルを読み込む
# ----------------------------------------------------------
# YOLO Pose は「人」を検出して, 17点の骨格座標を返す
# yolo11x は重いが検出が安定しやすい
pose_model = YOLO("yolo11x-pose.pt")

# ----------------------------------------------------------
# 3) 便利な関数たち（難しく見えるが, やってることは単純）
# ----------------------------------------------------------
def center_from_kpts(kp_xy):
    """
    人物の「中心」を推定する関数

    kp_xy は (17,2) で, 17個の点それぞれが (x,y)
    ただし検出できない点は NaN になっていることがある.

    やること:
    ・NaN じゃない点だけ使って平均を取り,
      だいたいの人物中心 (x,y) を作る

    返り値:
    ・中心 (x,y) の numpy配列
    ・全部 NaN なら None
    """
    valid = ~np.isnan(kp_xy).any(axis=1)  # xかyがNaNなら無効
    if not np.any(valid):
        return None
    return np.nanmean(kp_xy[valid], axis=0)


def inside_gate(x, y):
    """
    点(x,y)がベッド領域（gate_poly）の内側かどうかを調べる

    True:
      ベッドの中（患者の可能性が高い）
    False:
      ベッドの外（介助者の可能性が高い）
    """
    return cv2.pointPolygonTest(
        gate_poly,
        (float(x), float(y)),
        False
    ) >= 0


def compute_head_point(kp_xy, kp_c, conf_thr=0.3):
    """
    頭部の代表点 (x,y) を決める関数

    目的:
    ・後段解析で「頭の位置」を使うことがあるため

    優先順位:
    1) nose（鼻）が取れていれば鼻
    2) 鼻が無理なら, 目や耳（複数点）の平均

    kp_xy: (17,2) 座標
    kp_c : (17,) 各点の信頼度
    """
    # 0番は nose
    if kp_c[0] >= conf_thr and np.isfinite(kp_xy[0]).all():
        return kp_xy[0].copy(), "nose"

    # 目と耳の index（COCO形式）
    face_idx = [1, 2, 3, 4]

    valid = [
        i for i in face_idx
        if kp_c[i] >= conf_thr and np.isfinite(kp_xy[i]).all()
    ]

    if len(valid) > 0:
        return np.mean(kp_xy[valid], axis=0), "face"

    return np.array([np.nan, np.nan]), "none"


# 骨格の線を引くための「接続ルール」（見た目用）
SKELETON = [
    (0,1),(1,3),(0,2),(2,4),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]


def draw_skeleton(img, kp_xy, color=(0,0,255), th=2):
    """
    画像に骨格の線を描く（確認用）

    注意:
    ・解析のためではなく,
      「追跡が合ってるか」見るために描いている
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
# 4) 動画を開く + 出力動画の準備
# ----------------------------------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"VideoCapture failed: {video_path}")

# fps が取れない動画もあるので, その場合は 30fps とする
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

# 動画サイズ
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 骨格を描いた動画を書き出すための writer
out = cv2.VideoWriter(
    out_try_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H)
)

# ----------------------------------------------------------
# 5) 状態（state）の考え方
# ----------------------------------------------------------
# state は「今どういう状況か」を文字で持つ
#
# INIT:
#   まだ person1（患者役）が決まっていない
#
# LOCKED:
#   person1 が決まっていて追跡中
#
# MISSING:
#   そのフレームでベッド内に人が検出できない
#
# ※今のコードはシンプルに
#   毎フレーム「ベッド中心に近い人」を選ぶ設計になっている
state = "INIT"

person1_idx = None
prev_center = None

# ----------------------------------------------------------
# 6) フレームごとに保存したいデータの入れ物（リスト）
# ----------------------------------------------------------
# 後で np.array に変換して NPZ に入れる
all_kpts        = []   # 1フレームにつき (17,3) を保存（x,y,conf）
all_state       = []   # INIT / LOCKED / MISSING
all_idx         = []   # 選ばれた person の index（検出結果の何番目か）
all_head_xy     = []   # 頭代表点 (x,y)
all_head_source = []   # nose / face / none / missing

# ----------------------------------------------------------
# 7) 追跡用パラメータ
# ----------------------------------------------------------
# CONF_MIN:
#   これより信頼度が低い点は「無いもの」と扱って NaN にする
CONF_MIN = 0.3

# （今回は未使用. 連続性を厳密にするなら使う）
MAX_JUMP_PX = 120

# ----------------------------------------------------------
# 8) メインループ（1フレームずつ処理）
# ----------------------------------------------------------
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 動画が終わったら終了

    # ------------------------------------------
    # 8-1) YOLO Pose で検出
    # ------------------------------------------
    # res には:
    # ・boxes（人の四角い枠）
    # ・keypoints（骨格17点）
    # が入る
    res = pose_model.predict(frame, verbose=False, conf=CONF_MIN)[0]

    # 人の枠（使えない時もあるので保険付き）
    boxes = (
        res.boxes.xyxy.cpu().numpy()
        if res.boxes is not None else np.zeros((0,4))
    )

    # 骨格点:
    # shape は (人数, 17, 3)
    # 3 は (x, y, confidence)
    kpts = (
        res.keypoints.data.cpu().numpy()
        if res.keypoints is not None else np.zeros((0,17,3))
    )

    # ------------------------------------------
    # 8-2) 人が1人も検出されない場合
    # ------------------------------------------
    if kpts.shape[0] == 0:
        # 初回は NaN を入れておく
        # 2フレーム目以降は「直前データをそのまま入れる」設計
        all_kpts.append(
            all_kpts[-1] if frame_idx > 0 else np.full((17,3), np.nan)
        )
        all_state.append("MISSING")
        all_idx.append(-1)
        all_head_xy.append(np.array([np.nan, np.nan]))
        all_head_source.append("missing")
        frame_idx += 1
        continue

    # ------------------------------------------
    # 8-3) 「ベッド内にいる人」だけを候補にする
    # ------------------------------------------
    candidates = []

    # 検出された人数分だけループ
    for i in range(kpts.shape[0]):
        kp_xy = kpts[i, :, :2].copy()  # (17,2) 座標
        kp_c  = kpts[i, :, 2]          # (17,) 信頼度

        # 信頼度が低い点は NaN にする（後で無視できる）
        kp_xy[kp_c < CONF_MIN] = np.nan

        # 骨格から中心を推定
        center = center_from_kpts(kp_xy)

        # 骨格が全滅で中心が作れなければ, box中心で代用
        if center is None:
            x1, y1, x2, y2 = boxes[i]
            center = np.array([(x1+x2)/2, (y1+y2)/2])

        # ベッド内なら候補に追加
        if inside_gate(center[0], center[1]):
            candidates.append(
                dict(i=i, center=center, kp_xy=kp_xy, kp_c=kp_c)
            )

    # 候補が誰もいなければ MISSING 扱い
    if not candidates:
        all_kpts.append(all_kpts[-1])
        all_state.append("MISSING")
        all_idx.append(-1)
        all_head_xy.append(np.array([np.nan, np.nan]))
        all_head_source.append("missing")
        frame_idx += 1
        continue

    # ------------------------------------------
    # 8-4) 候補の中から「ベッド中心に一番近い人」を選ぶ
    # ------------------------------------------
    # これが person1（患者役）になる
    chosen = min(
        candidates,
        key=lambda c: np.hypot(
            c["center"][0] - bed_cx,
            c["center"][1] - bed_cy
        )
    )

    # 追跡状態を LOCKED にする
    state = "LOCKED"
    person1_idx = chosen["i"]
    prev_center = chosen["center"]

    # ------------------------------------------
    # 8-5) 保存用の形に整形して記録
    # ------------------------------------------
    # chosen["kp_xy"] は (17,2)
    # chosen["kp_c"]  は (17,)
    #
    # 保存したいのは (17,3) つまり (x,y,conf)
    row_k = np.stack(
        [chosen["kp_xy"][:, 0], chosen["kp_xy"][:, 1], chosen["kp_c"]],
        axis=1
    )

    # 頭の代表点も計算して保存する
    head_xy, head_src = compute_head_point(
        chosen["kp_xy"],
        chosen["kp_c"],
        CONF_MIN
    )

    # 各リストに追加（フレームごと）
    all_kpts.append(row_k)
    all_state.append(state)
    all_idx.append(person1_idx)
    all_head_xy.append(head_xy)
    all_head_source.append(head_src)

    # ------------------------------------------
    # 8-6) 確認用に骨格を描いて動画に保存
    # ------------------------------------------
    draw_skeleton(frame, chosen["kp_xy"])
    out.write(frame)

    frame_idx += 1

# ----------------------------------------------------------
# 9) 終了処理（動画を閉じる）
# ----------------------------------------------------------
cap.release()
out.release()

# ----------------------------------------------------------
# 10) NPZ 保存のために numpy配列に変換
# ----------------------------------------------------------
kpts_raw  = np.array(all_kpts)          # shape: (フレーム数, 17, 3)
state_arr = np.array(all_state)         # shape: (フレーム数,)
time_all  = np.arange(len(kpts_raw)) / fps  # 各フレームの時刻（秒）

# ----------------------------------------------------------
# 11) NPZ（numpy形式の保存ファイル）に書き出す
# ----------------------------------------------------------
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
# 12) QuickTime で再生できるように ffmpeg で再エンコード
# ----------------------------------------------------------
# OpenCVで書いた mp4 は, MacのQuickTimeで再生できないことがある.
# そのため ffmpeg で「互換性が高い形式」に変換する.
os.system(
    f"ffmpeg -y -i {out_try_path} "
    f"-vcodec libx264 -pix_fmt yuv420p "
    f"-profile:v baseline -level 3.0 "
    f"-movflags +faststart {out_qt_path}"
)

print("[OK] QT video:", out_qt_path)
