"""
person1_tracker_beginner.py

目的.
  VEEGの動画から, ベッド上の「患者さん(person1)」を安定して追跡し,
  YOLO Pose (17点骨格) の時系列をNPZに保存する.

このスクリプトの考え方(重要).
  1) 各フレームでYOLO Poseを実行して, 人ごとの骨格(17点)を得る.
  2) 骨格の中心(=有効点の平均)が, ベッドの多角形(gate)の内側にある人だけを候補にする.
  3) 候補が「ちょうど1人」なら, その人を person1 としてLOCKする.
  4) 候補が0人または2人以上なら, 直前のLOCK結果を使う(carry)か, NaNにする.

注意.
  - gate_polygon_xy (N,2) は, ユーザーが事前に用意して渡す.
  - モデル重み(model_path)もユーザーが用意する.
"""

from __future__ import annotations

import numpy as np
import cv2
from ultralytics import YOLO


# ==========================================================
# 1) 17点骨格の「線のつなぎ方」(COCO形式)
#    これはQC動画で線を描くときだけ使う. 解析には必須ではない.
# ==========================================================
COCO_SKELETON = [
    (0, 1), (1, 3), (0, 2), (2, 4),        # 顔周り
    (5, 7), (7, 9), (6, 8), (8, 10),       # 腕
    (5, 6), (5, 11), (6, 12), (11, 12),    # 体幹
    (11, 13), (13, 15), (12, 14), (14, 16) # 脚
]


# ==========================================================
# 2) 小さな便利関数(初心者向けに, 役割を小さく保つ)
# ==========================================================
def to_gate_poly(gate_polygon_xy: np.ndarray) -> np.ndarray:
    """
    (N,2) の多角形座標を, OpenCVが扱いやすい形 (N,1,2) に変換する.

    gate_polygon_xy: 画像座標の点列. 例: [[x0,y0],[x1,y1],...]
    """
    gate_polygon_xy = np.asarray(gate_polygon_xy, dtype=np.float32)

    if gate_polygon_xy.ndim != 2 or gate_polygon_xy.shape[1] != 2:
        raise ValueError("gate_polygon_xy must be shaped (N, 2).")

    return gate_polygon_xy.reshape(-1, 1, 2)


def point_inside_polygon(gate_poly: np.ndarray, x: float, y: float) -> bool:
    """
    点(x,y)が, 多角形の内側にあるかを判定する.

    OpenCVの pointPolygonTest を使う.
    戻り値 >= 0 なら内側(境界含む).
    """
    v = cv2.pointPolygonTest(gate_poly, (float(x), float(y)), False)
    return v >= 0


def skeleton_center(kp_xy: np.ndarray) -> np.ndarray | None:
    """
    骨格17点のうち, 有効な点(=NaNでない点)の平均を「中心」として返す.

    kp_xy: (17,2) , 欠損はNaN.
    戻り値: (2,) , もし有効点がゼロなら None.
    """
    valid = np.isfinite(kp_xy).all(axis=1)
    if not np.any(valid):
        return None
    return np.nanmean(kp_xy[valid], axis=0)


def draw_skeleton(frame_bgr: np.ndarray, kp_xy: np.ndarray, thickness: int = 2) -> None:
    """
    QC用. 画像(frame)に骨格の線を描く.
    """
    for a, b in COCO_SKELETON:
        xa, ya = kp_xy[a]
        xb, yb = kp_xy[b]

        if np.isfinite([xa, ya, xb, yb]).all():
            cv2.line(
                frame_bgr,
                (int(xa), int(ya)),
                (int(xb), int(yb)),
                (0, 0, 255),
                thickness
            )


# ==========================================================
# 3) YOLO Poseを「1フレームだけ」実行する関数
# ==========================================================
def run_yolo_pose(model: YOLO, frame_bgr: np.ndarray, conf_min: float) -> tuple[np.ndarray, np.ndarray]:
    """
    1フレームでYOLO Pose推論をする.

    戻り値.
      boxes: (N,4) , [x1,y1,x2,y2]
      kpts_all: (N,17,3) , [x,y,conf]
    """
    result = model.predict(frame_bgr, verbose=False, conf=conf_min)[0]

    # boxが無い場合もあるので, その時は空配列にする.
    if result.boxes is None:
        boxes = np.zeros((0, 4), dtype=float)
    else:
        boxes = result.boxes.xyxy.cpu().numpy()

    # keypointsが無い場合もあるので, その時は空配列にする.
    if result.keypoints is None:
        kpts_all = np.zeros((0, 17, 3), dtype=float)
    else:
        kpts_all = result.keypoints.data.cpu().numpy()

    return boxes, kpts_all


# ==========================================================
# 4) gate内にいる「候補者」を集める
# ==========================================================
def collect_candidates(
    boxes: np.ndarray,
    kpts_all: np.ndarray,
    gate_poly: np.ndarray,
    conf_min: float
) -> list[dict]:
    """
    gate内にいる人を候補として集める.

    方針.
      - keypointのconfが低い点はNaN扱いにする.
      - 骨格中心(有効点の平均)が取れればそれを使う.
      - 取れなければbbox中心を使う.
      - その中心がgate内なら候補に入れる.

    戻り値.
      candidates: 各要素は dict
        {
          "idx": YOLO出力の人index,
          "kp_xy": (17,2),
          "kp_c": (17,),
          "center_xy": (2,)
        }
    """
    candidates: list[dict] = []

    for i in range(kpts_all.shape[0]):
        kp_xy = kpts_all[i, :, :2].copy()   # (17,2)
        kp_c  = kpts_all[i, :, 2].copy()    # (17,)

        # confが低い点は欠損(NaN)にする.
        kp_xy[kp_c < conf_min] = np.nan

        # まず骨格中心.
        center = skeleton_center(kp_xy)

        # 骨格中心が取れないならbbox中心にする.
        if center is None:
            if boxes.shape[0] <= i:
                continue
            x1, y1, x2, y2 = boxes[i]
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)

        # gate内にいれば候補.
        if point_inside_polygon(gate_poly, center[0], center[1]):
            candidates.append(
                dict(idx=i, kp_xy=kp_xy, kp_c=kp_c, center_xy=center)
            )

    return candidates


def to_row_k(kp_xy: np.ndarray, kp_c: np.ndarray) -> np.ndarray:
    """
    保存用に(17,3)へまとめる. [x,y,conf]
    """
    row_k = np.stack([kp_xy[:, 0], kp_xy[:, 1], kp_c], axis=1)
    return row_k.astype(float)


# ==========================================================
# 5) メイン関数: 追跡してNPZ保存
# ==========================================================
def track_person1_and_save(
    video_path: str,
    model_path: str,
    gate_polygon_xy: np.ndarray,
    out_npz_path: str,
    conf_min: float = 0.3,
    carry_forward: bool = True,
    qc_video_path: str | None = None
) -> None:
    """
    person1追跡を実行し, NPZに保存する.

    保存されるNPZ.
      kpts_raw: (N_frames,17,3)
      fps: float
      time_all: (N_frames,)
      state: (N_frames,) 文字列 "LOCKED" "CARRY" "MISSING" "AMBIGUOUS"
      person1_idx: (N_frames,) int (LOCKした時のYOLO index, それ以外は-1)
      gate_polygon_xy: (N,2)
    """
    gate_poly = to_gate_poly(gate_polygon_xy)

    # 動画オープン.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"VideoCapture failed: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # YOLOモデル読込.
    model = YOLO(model_path)

    # QC動画出力(オプション).
    writer = None
    if qc_video_path is not None:
        writer = cv2.VideoWriter(
            qc_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (W, H)
        )

    all_kpts: list[np.ndarray] = []
    all_state: list[str] = []
    all_idx: list[int] = []

    last_valid_row_k: np.ndarray | None = None

    # 1フレームずつ処理.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, kpts_all = run_yolo_pose(model, frame, conf_min)

        # 1) 人が検出されない.
        if kpts_all.shape[0] == 0:
            if carry_forward and (last_valid_row_k is not None):
                row_k = last_valid_row_k.copy()
                state = "CARRY"
            else:
                row_k = np.full((17, 3), np.nan, dtype=float)
                state = "MISSING"
            idx = -1

        else:
            # 2) gate内の候補を集める.
            candidates = collect_candidates(boxes, kpts_all, gate_poly, conf_min)

            # 2a) 候補がちょうど1人 -> LOCK.
            if len(candidates) == 1:
                c = candidates[0]
                row_k = to_row_k(c["kp_xy"], c["kp_c"])
                state = "LOCKED"
                idx = int(c["idx"])
                last_valid_row_k = row_k

                # QC動画があるなら骨格を描く.
                if writer is not None:
                    draw_skeleton(frame, c["kp_xy"], thickness=2)

            # 2b) 候補が0人, または2人以上 -> carry or NaN.
            else:
                if carry_forward and (last_valid_row_k is not None):
                    row_k = last_valid_row_k.copy()
                    state = "CARRY"
                else:
                    row_k = np.full((17, 3), np.nan, dtype=float)
                    state = "MISSING"

                if len(candidates) >= 2:
                    state = "AMBIGUOUS"
                idx = -1

        all_kpts.append(row_k)
        all_state.append(state)
        all_idx.append(idx)

        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    # 保存.
    kpts_raw = np.asarray(all_kpts, dtype=float)  # (N,17,3)
    time_all = np.arange(kpts_raw.shape[0], dtype=float) / float(fps)

    np.savez(
        out_npz_path,
        kpts_raw=kpts_raw,
        fps=float(fps),
        time_all=time_all,
        state=np.asarray(all_state, dtype=object),
        person1_idx=np.asarray(all_idx, dtype=int),
        gate_polygon_xy=np.asarray(gate_polygon_xy, dtype=float),
        model_path=np.asarray(model_path, dtype=object),
        video_path=np.asarray(video_path, dtype=object),
        conf_min=float(conf_min),
        carry_forward=bool(carry_forward),
    )


# ==========================================================
# 6) 使い方例 (このまま動かす場合)
# ==========================================================
if __name__ == "__main__":
    # 例: gate_polygon_xyを自分で用意する.
    # 実データの座標を入れる. 下はダミー例.
    gate_polygon_xy = np.array([
        [100, 100],
        [500, 120],
        [520, 380],
        [120, 400],
    ], dtype=float)

    track_person1_and_save(
        video_path="input.mp4",
        model_path="yolo11x-pose.pt",
        gate_polygon_xy=gate_polygon_xy,
        out_npz_path="track_person1.npz",
        conf_min=0.3,
        carry_forward=True,
        qc_video_path="track_person1_qc.mp4",  # QC不要なら None
    )

    print("Done. NPZ saved.")
