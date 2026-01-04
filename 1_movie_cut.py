# ==========================================================
# mp4動画を「○秒から○秒まで」切り出すコード
#
# ポイント:
# ・動画は「秒」ではなく「フレーム番号」で扱われる
# ・そのため「秒 → フレーム番号」に変換して処理している
# ・元の動画は消さず, 新しい動画ファイルを作る
# ==========================================================

import cv2, os   # 動画処理(cv2)とファイル操作(os)を使う

# ----------------------------------------------------------
# ① ここだけ変更すればOKな設定
# ----------------------------------------------------------

# 元の動画ファイル
SRC_MP4 = '/content/FBTCS.mp4'

# 切り出した動画の保存先
OUT_MP4 = '/content/FBTCS_qt.mp4'

# 切り出し開始時間（秒）
START_SEC = 13

# 切り出し終了時間（秒）
END_SEC = 28     # None にすると最後まで

# ----------------------------------------------------------
# ② 元の動画ファイルが存在するか確認
# ----------------------------------------------------------
if not os.path.exists(SRC_MP4):
    # ファイルが無い場合はここで止める
    raise FileNotFoundError("動画ファイルが見つかりません")

# ----------------------------------------------------------
# ③ 動画ファイルを開く
# ----------------------------------------------------------
cap = cv2.VideoCapture(SRC_MP4)

if not cap.isOpened():
    # 動画が開けない場合はエラー
    raise RuntimeError("動画を開けませんでした")

# ----------------------------------------------------------
# ④ 動画の基本情報を取得
# ----------------------------------------------------------

# fps = 1秒あたり何フレームあるか
fps = cap.get(cv2.CAP_PROP_FPS)

# 動画の横と縦の大きさ（ピクセル）
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("動画サイズを取得できません")

print(f"fps = {fps}")
print(f"size = {width} x {height}")

# ----------------------------------------------------------
# ⑤ 「秒」を「フレーム番号」に変換
# ----------------------------------------------------------
# 動画はフレーム番号で扱うため,
# 秒 × fps = フレーム番号 になる

start_frame = int(round(START_SEC * fps))
end_frame   = int(round(END_SEC * fps)) if END_SEC is not None else None

# マイナスにならないように保険
start_frame = max(0, start_frame)
if end_frame is not None:
    end_frame = max(0, end_frame)

print(f"切り出し: frame {start_frame} 〜 {end_frame}")

# ----------------------------------------------------------
# ⑥ 出力用の動画ファイルを準備
# ----------------------------------------------------------
# いきなり完成ファイルを書かず,
# 一時ファイルに書いてから置き換える（安全のため）

tmp_mp4 = OUT_MP4 + '.__tmp__.mp4'

if os.path.exists(tmp_mp4):
    os.remove(tmp_mp4)

# mp4用の一般的な形式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 書き込み用オブジェクトを作成
out = cv2.VideoWriter(tmp_mp4, fourcc, fps, (width, height))

if not out.isOpened():
    cap.release()
    raise RuntimeError("出力動画を作れませんでした")

# ----------------------------------------------------------
# ⑦ 開始フレームまで移動
# ----------------------------------------------------------
# 指定したフレーム番号まで移動する
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# ----------------------------------------------------------
# ⑧ フレームを1枚ずつコピーする
# ----------------------------------------------------------
current_frame = start_frame
written = 0

while True:
    # 終了フレームに到達したら止める
    if end_frame is not None and current_frame >= end_frame:
        break

    # フレームを1枚読む
    ret, frame = cap.read()

    if not ret:
        # 読めなければ動画の終わり
        break

    # フレームを書き込む
    out.write(frame)

    written += 1
    current_frame += 1

# ----------------------------------------------------------
# ⑨ 後片付け
# ----------------------------------------------------------
cap.release()
out.release()

if written == 0:
    if os.path.exists(tmp_mp4):
        os.remove(tmp_mp4)
    raise RuntimeError("フレームを書き出せませんでした")

# 一時ファイルを正式な出力ファイルに変更
os.replace(tmp_mp4, OUT_MP4)

print("動画の切り出しが完了しました")
