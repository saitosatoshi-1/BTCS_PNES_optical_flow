# ==========================================================
# Trim an MP4 video using time (in seconds)
#
# This cell:
#   - extracts a video segment from START_SEC to END_SEC
#   - keeps the original video unchanged
#   - saves the trimmed segment as a new MP4 file
#
# This is useful when you want to analyze only a specific
# seizure segment from a long VEEG recording.
# ==========================================================

import cv2
import os

# ----------------------------------------------------------
# User settings
# ----------------------------------------------------------
# SRC_MP4   : path to the input video file
# OUT_MP4   : path to the output (trimmed) video file
# START_SEC : start time in seconds
# END_SEC   : end time in seconds
#             set to None to extract until the end of the video
SRC_MP4   = "/content/FBTCS.mp4"
OUT_MP4   = "/content/FBTCS_qt.mp4"
START_SEC = 19     # start at 19 seconds
END_SEC   = 34     # end at 34 seconds (None = until EOF)

# ----------------------------------------------------------
# Check that the input video exists
# ----------------------------------------------------------
if not os.path.exists(SRC_MP4):
    raise FileNotFoundError(f"Input video not found: {SRC_MP4}")

# Open the video using OpenCV
cap = cv2.VideoCapture(SRC_MP4)
if not cap.isOpened():
    raise RuntimeError("Failed to open video with VideoCapture")

# ----------------------------------------------------------
# Read basic video properties
# ----------------------------------------------------------
fps = cap.get(cv2.CAP_PROP_FPS)

# Total number of frames (may be unavailable for some codecs)
frame_count_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = frame_count_raw if frame_count_raw > 0 else None

# Frame size
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Safety check
if width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError(
        "Failed to read video size (corrupted video or unsupported codec)"
    )

print(
    f"[INFO] fps={fps:.3f}, "
    f"frames={frame_count if frame_count is not None else 'unknown'}, "
    f"size={width}x{height}"
)

# ----------------------------------------------------------
# Convert time (seconds) to frame indices
# ----------------------------------------------------------
# Example:
#   15.0 seconds × 30 fps = frame 450
#
# round() is used to select the nearest frame.
start_frame = int(round(START_SEC * fps))
end_frame   = None if END_SEC is None else int(round(END_SEC * fps))

# Log trimming range
if end_frame is None:
    print(
        f"[INFO] trim frames: {start_frame} .. EOF "
        f"({start_frame / fps:.3f} sec ..)"
    )
else:
    print(
        f"[INFO] trim frames: {start_frame} .. {end_frame - 1} "
        f"({start_frame / fps:.3f}–{(end_frame - 1) / fps:.3f} sec)"
    )

# ----------------------------------------------------------
# Seek to the starting frame
# ----------------------------------------------------------
# Note:
# cap.set() may silently fail for some video formats,
# so we always check the actual frame position.
ok = cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
print(
    f"[INFO] seek requested={start_frame}, "
    f"actual_pos={pos}, ok={ok}"
)

# ----------------------------------------------------------
# Prepare a temporary output file
# ----------------------------------------------------------
tmp_mp4 = OUT_MP4 + ".__tmp__.mp4"

# Remove existing temporary file if present
if os.path.exists(tmp_mp4):
    os.remove(tmp_mp4)

# Use a common MP4 codec
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Create the video writer
out = cv2.VideoWriter(tmp_mp4, fourcc, fps, (width, height))
if not out.isOpened():
    cap.release()
    raise RuntimeError(f"Failed to open VideoWriter: {tmp_mp4}")

# ----------------------------------------------------------
# Read and write frames one by one
# ----------------------------------------------------------
written = 0

# Maximum number of frames to write (if END_SEC is specified)
if end_frame is None:
    max_to_write = None
else:
    max_to_write = max(0, end_frame - start_frame)

while True:
    # Stop if we reached the desired length
    if max_to_write is not None and written >= max_to_write:
        break

    ret, frame = cap.read()
    if not ret:
        print(
            "[WARNING] Failed to read frame "
            "(end of video or seek failure)"
        )
        break

    out.write(frame)
    written += 1

cap.release()
out.release()

# ----------------------------------------------------------
# Check that at least one frame was written
# ----------------------------------------------------------
if written <= 0:
    if os.path.exists(tmp_mp4):
        os.remove(tmp_mp4)
    raise RuntimeError(
        "No frames were written. "
        "Please check the input video and time range."
    )

# ----------------------------------------------------------
# Replace the output file atomically
# ----------------------------------------------------------
# os.replace is safer than os.rename if something goes wrong
os.replace(tmp_mp4, OUT_MP4)

print(f"[OK] Trimmed video saved: {OUT_MP4}")
print(f"[OK] Original video kept unchanged: {SRC_MP4}")
