""" Interactive BED polygon editor (Jupyter)
This cell helps you define a 4-point polygon (bed/ROI gate) on a representative video frame.
You can adjust each vertex and apply a global rotation, then copy the printed coordinates
into a JSON configuration file (e.g., `gate_config.json`).

Note: This UI uses `ipywidgets`, so it works best in Jupyter/Colab.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider

# ==========================================================
# Goal
#   Display a single video frame and overlay a 4-point polygon (a rectangle-like ROI).
#   You can adjust each vertex (x, y) with sliders and also rotate the polygon.
#
# Typical use case
#   Create polygon coordinates for a gate_config.json (or similar config file),
#   so that "person1" tracking can be restricted to the bed area.
#
# Notes for beginners
#   - OpenCV reads images in BGR color order, but matplotlib expects RGB.
#   - A polygon is just a list of (x, y) points ordered around the shape.
#   - Rotation is done around the polygon center (centroid).
# ==========================================================

# ----------------------------------------------------------
# Settings you may want to change
# ----------------------------------------------------------
# best_frame_idx:
#   Choose a frame where the bed and patient are clearly visible.
best_frame_idx = 30

# Path to the (trimmed) MP4 video
video_path = "/content/FBTCS_qt.mp4"

# ----------------------------------------------------------
# Load one frame from the video
# ----------------------------------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"VideoCapture failed: {video_path}")

# Jump to the target frame index
cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to load the requested frame. Check best_frame_idx.")

# Frame size (height H, width W)
H, W = frame.shape[:2]

# ----------------------------------------------------------
# Initial polygon (a rough rectangle)
#   Point order: [top-left, top-right, bottom-right, bottom-left]
# ----------------------------------------------------------
tL = np.array([W * 0.25, H * 0.30])  # top-left
tR = np.array([W * 0.75, H * 0.30])  # top-right
bR = np.array([W * 0.75, H * 0.70])  # bottom-right
bL = np.array([W * 0.25, H * 0.70])  # bottom-left

# Optional: keep the latest rotated polygon in a global variable
# (useful if you want to access it in another cell)
poly_rot_global = None

# ----------------------------------------------------------
# Update function (called every time a slider changes)
# ----------------------------------------------------------
def update(
    tL_x, tL_y,
    tR_x, tR_y,
    bR_x, bR_y,
    bL_x, bL_y,
    rotate_deg
):
    """
    Build a polygon from slider values, rotate it around its center,
    and draw it on top of the selected frame.
    """
    global poly_rot_global

    # 1) Build the polygon from slider inputs (float for math)
    poly = np.array(
        [
            [tL_x, tL_y],
            [tR_x, tR_y],
            [bR_x, bR_y],
            [bL_x, bL_y],
        ],
        dtype=float
    )

    # 2) Rotation around the polygon center (centroid)
    # Convert degrees to radians
    theta = np.deg2rad(rotate_deg)

    # 2D rotation matrix
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ],
        dtype=float
    )

    # Center of the polygon (mean of vertex coordinates)
    center = poly.mean(axis=0)

    # Translate to origin -> rotate -> translate back
    poly_rot = (poly - center) @ R.T + center

    # Save the latest polygon for later use
    poly_rot_global = poly_rot.copy()

    # 3) Draw polygon on the frame
    img = frame.copy()

    cv2.polylines(
        img,
        [poly_rot.astype(int)],
        isClosed=True,
        color=(0, 0, 255),  # red in BGR
        thickness=3
    )

    # OpenCV is BGR, matplotlib expects RGB
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Adjust BED polygon (with rotation)")
    plt.axis("off")
    plt.show()

    # 4) Print polygon coordinates (copy/paste into JSON)
    print("CURRENT ROTATED POLYGON COORDS (x, y):")
    print(poly_rot)

# ----------------------------------------------------------
# Interactive sliders
#   - Each vertex can be adjusted independently
#   - rotate_deg rotates the whole polygon
# ----------------------------------------------------------
interact(
    update,
    tL_x=IntSlider(min=0, max=W, value=int(tL[0]), step=2, description="tL_x"),
    tL_y=IntSlider(min=0, max=H, value=int(tL[1]), step=2, description="tL_y"),

    tR_x=IntSlider(min=0, max=W, value=int(tR[0]), step=2, description="tR_x"),
    tR_y=IntSlider(min=0, max=H, value=int(tR[1]), step=2, description="tR_y"),

    bR_x=IntSlider(min=0, max=W, value=int(bR[0]), step=2, description="bR_x"),
    bR_y=IntSlider(min=0, max=H, value=int(bR[1]), step=2, description="bR_y"),

    bL_x=IntSlider(min=0, max=W, value=int(bL[0]), step=2, description="bL_x"),
    bL_y=IntSlider(min=0, max=H, value=int(bL[1]), step=2, description="bL_y"),

    rotate_deg=FloatSlider(min=-45, max=45, step=1, value=0, description="Rotate"),
)
