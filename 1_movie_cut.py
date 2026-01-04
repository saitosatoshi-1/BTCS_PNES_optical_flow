# ==========================================================
# mp4動画を「時間（秒）」で切り出すセル
#
# できること:
#   ・動画の START_SEC 秒から END_SEC 秒までを切り出す
#   ・元動画は消さず, 切り出した動画だけを新しく保存する
#
# 使い方:
#   1. SRC_MP4, OUT_MP4 を自分の環境に合わせて変更
#   2. START_SEC, END_SEC を秒で指定
#   3. セルを実行
# ==========================================================

import cv2, os

# ----------------------------------------------------------
# このセルの目的
# ----------------------------------------------------------
# SRC_MP4 にある元動画から,
# 「START_SEC 秒〜 END_SEC 秒」の区間だけを取り出して
# OUT_MP4 という新しい動画として保存します.
#
# 途中でエラーが起きても OUT_MP4 が壊れないように,
# いったん「一時ファイル」に書き出してから置き換える
# 安全な方法を使っています.
# ----------------------------------------------------------

# ---- 設定（ここだけ書き換えればOK）----
# SRC_MP4   : 入力動画のパス
# OUT_MP4   : 出力動画のパス（切り出し後）
# START_SEC : 切り出し開始時間（秒）
# END_SEC   : 切り出し終了時間（秒）
#             None にすると「最後まで」
SRC_MP4    = '/content/FBTCS.mp4'
OUT_MP4    = '/content/FBTCS_qt.mp4'
START_SEC  = 13    # 15秒から
END_SEC    = 28   # 25秒まで（Noneなら最後まで）

# ----------------------------------------------------------
# 入力動画が本当に存在するか確認
# ----------------------------------------------------------
if not os.path.exists(SRC_MP4):
    raise FileNotFoundError(f'入力動画が見つかりません: {SRC_MP4}')

# OpenCVで動画を開く
cap = cv2.VideoCapture(SRC_MP4)
if not cap.isOpened():
    raise RuntimeError('VideoCaptureが開けません')

# ----------------------------------------------------------
# 動画の基本情報を取得
# ----------------------------------------------------------
# fps（1秒あたりのフレーム数）
# 取得できない環境もあるので, その場合は30fpsと仮定
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

# 総フレーム数（不明な場合もある）
frame_count_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = frame_count_raw if frame_count_raw > 0 else None

# 動画サイズ（幅×高さ）
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 動画サイズが取得できない場合はエラー
if width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("動画サイズを取得できません（破損動画や非対応コーデックの可能性）")

print(f"[INFO] src: {SRC_MP4}")
print(f"       fps={fps:.3f}, frames={frame_count if frame_count is not None else 'unknown'}, size={width}x{height}")
print(f"       out: {OUT_MP4}")

# ----------------------------------------------------------
# 「秒」→「フレーム番号」に変換
# ----------------------------------------------------------
# 例:
#   15.0 秒 × 30 fps = 450 フレーム目
# round() で最も近いフレームに合わせます
start_frame = int(round(START_SEC * fps))
end_frame   = None if END_SEC is None else int(round(END_SEC * fps))

# マイナスにならないように補正
start_frame = max(0, start_frame)
if end_frame is not None:
    end_frame = max(0, end_frame)

# 総フレーム数が分かる場合は, はみ出さないように調整
if frame_count is not None:
    start_frame = min(start_frame, frame_count)
    if end_frame is None:
        end_frame = frame_count
    else:
        end_frame = min(end_frame, frame_count)

# 開始より終了が前になっていないかチェック
if end_frame is not None and end_frame <= start_frame:
    cap.release()
    raise ValueError(f"切り出し範囲が不正です: start_frame={start_frame}, end_frame={end_frame}")

# 切り出す範囲をログ表示
if end_frame is None:
    print(f"[INFO] trim frames: {start_frame} .. EOF  ({start_frame/fps:.3f} sec ..)")
else:
    print(f"[INFO] trim frames: {start_frame} .. {end_frame-1}  "
          f"({start_frame/fps:.3f}–{(end_frame-1)/fps:.3f} sec)")

# ----------------------------------------------------------
# 一時ファイルに動画を書き出す
# ----------------------------------------------------------
# いきなり OUT_MP4 に書かず,
# OUT_MP4.__tmp__.mp4 に書いてから置き換えます
tmp_mp4 = OUT_MP4 + '.__tmp__.mp4'
if os.path.exists(tmp_mp4):
    os.remove(tmp_mp4)

# mp4用の一般的なコーデックを指定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 出力動画を作成
out = cv2.VideoWriter(tmp_mp4, fourcc, fps, (width, height))
if not out.isOpened():
    cap.release()
    raise RuntimeError(f'VideoWriterが開けません: {tmp_mp4}')

# ----------------------------------------------------------
# 開始フレームまで移動（シーク）
# ----------------------------------------------------------
# cv2.CAP_PROP_POS_FRAMES を用いたシークは、成功を報告しても
# 実際にその後の cap.read() が失敗することがあります。
# より堅牢な方法として、必要に応じてVideoCaptureを再オープンし、
# 0フレーム目から目的の開始フレームまでフレームを読み飛ばす処理を追加します。

final_cap_obj = None

# まずは通常のシークを試みる
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
ret_after_seek, _ = cap.read()

if ret_after_seek:
    # シーク後、最初のフレームが正常に読めた場合、そのまま続行
    # ただし、`cap.read()`で1フレーム進んだので、`start_frame`に戻す
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    final_cap_obj = cap
else:
    # シークが失敗したか、シーク後すぐに読み込みが失敗した場合
    print(f"[WARNING] 直接シーク (start_frame={start_frame}) が失敗したか、直後の読み込みができませんでした。"
          "Video Captureを再初期化し、0フレームから順に読み飛ばしてシークします。")

    # 既存の cap を解放して再オープン
    cap.release()
    cap = cv2.VideoCapture(SRC_MP4)
    if not cap.isOpened():
        out.release()
        if os.path.exists(tmp_mp4): os.remove(tmp_mp4)
        raise RuntimeError(f"Video Captureを再オープンできませんでした: {SRC_MP4}")

    # 0フレーム目から開始フレームまで読み飛ばす
    for i in range(start_frame):
        ret_skip, _ = cap.read()
        if not ret_skip:
            cap.release()
            out.release()
            if os.path.exists(tmp_mp4): os.remove(tmp_mp4)
            raise RuntimeError(f"開始フレーム ({start_frame}) までのスキップ中にフレーム {i} の読み込みに失敗しました。動画の破損や読み取り不能区間の可能性があります。")
    final_cap_obj = cap # 読み飛ばしが成功したら、このcapオブジェクトを使用

if final_cap_obj is None:
    cap.release()
    out.release()
    if os.path.exists(tmp_mp4): os.remove(tmp_mp4)
    raise RuntimeError("動画の開始フレームへの配置に失敗しました。処理を中断します。")

# ここから、final_cap_obj を `cap` として使用する
cap = final_cap_obj

# ----------------------------------------------------------
# フレームを1枚ずつ読み込んで書き出し
# ----------------------------------------------------------
cur = start_frame
written = 0

while True:
    # 終了フレームに到達したら止める
    if end_frame is not None and cur >= end_frame:
        break

    ret, frame = cap.read()
    if not ret:
        # ここで読み込みに失敗する場合は、動画の範囲外または破損
        print(f"[WARNING] フレーム {cur} の読み込みに失敗しました (ret=False)。動画の終端か破損の可能性があります。")
        break

    out.write(frame)   # フレームを書き込み
    written += 1
    cur += 1

# リソース解放
cap.release()
out.release()

# ----------------------------------------------------------
# 1フレームも書けていない場合はエラー
# ----------------------------------------------------------
if written <= 0:
    if os.path.exists(tmp_mp4):
        os.remove(tmp_mp4)
    raise RuntimeError('フレームを書き出せませんでした（入力動画/範囲を確認してください）')

# ----------------------------------------------------------
# 一時ファイルを正式な出力ファイルに置き換え
# ----------------------------------------------------------
# os.replace は途中失敗でもファイルが壊れにくい安全な方法
os.replace(tmp_mp4, OUT_MP4)

print(f"[OK] Trimmed: {OUT_MP4} (kept source: {SRC_MP4})")
