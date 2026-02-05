# viz.py
from __future__ import annotations

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# -----------------------------
# Robust asset paths (relative to this file)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAR_RIGHT = os.path.join(BASE_DIR, "car_right.png")   # faces right
CAR_LEFT  = os.path.join(BASE_DIR, "car_left.png")    # faces left
PED_WAIT  = os.path.join(BASE_DIR, "people.png")      # standing (curb + median waiting)
PED_WALK  = os.path.join(BASE_DIR, "walk.png")        # walking (while crossing)


def _load_img(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing image: {path}")
    return plt.imread(path)


IMG_CAR_RIGHT = _load_img(CAR_RIGHT)
IMG_CAR_LEFT  = _load_img(CAR_LEFT)
IMG_PED_WAIT  = _load_img(PED_WAIT)
IMG_PED_WALK  = _load_img(PED_WALK)


def draw_png(ax, img, x, y, zoom=0.08, zorder=10):
    image = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(image, (x, y), frameon=False, box_alignment=(0.5, 0.5), zorder=zorder)
    ax.add_artist(ab)


def visualize_step(
    env_data: dict,
    spawn_distance: float = 30.0,  # <-- zoom IN by default (fixes "ped looks faster")
    dt: float = 0.2,               # <-- pause per frame to match simulation time
    equal_aspect: bool = False,    # True = 1m on x equals 1m on y (may look very wide)
    show_title: bool = True,
):
    """
    Draw one frame from env.render_data().

    Expected env_data keys (your env.render_data()):
      - cars_lane1: list[(x, v)]  (lane 0, right->left)
      - cars_lane2: list[(x, v)]  (lane 1, left->right)
      - ped_stage: int (0 curb, 1 cross lane1, 2 median, 3 cross lane2)
      - stage_progress: float in [0,1] (only while crossing)
      - terminal_event: optional str ("collision"/"success"/"timeout") if you add it
    """

    plt.clf()
    ax = plt.gca()

    # -----------------------------
    # Geometry (meters)
    # -----------------------------
    road_h = 7.0
    curb_w = 0.5
    road_y0 = 0.0

    lane_w = road_h / 2.0
    lane2_y = road_y0 + lane_w / 2.0          # lower lane center
    lane1_y = road_y0 + 1.5 * lane_w          # upper lane center
    center_line_y = road_y0 + road_h / 2.0

    bottom_sidewalk_y = road_y0 - curb_w / 2.0
    top_sidewalk_y = road_y0 + road_h + curb_w / 2.0

    # "Median" waiting position: between lanes (center line)
    median_y = center_line_y

    # -----------------------------
    # Background
    # -----------------------------
    # bottom sidewalk
    ax.add_patch(Rectangle(
        (-spawn_distance, road_y0 - curb_w),
        2 * spawn_distance, curb_w,
        facecolor="#DDDDDD", edgecolor="none", zorder=0
    ))

    # road
    ax.add_patch(Rectangle(
        (-spawn_distance, road_y0),
        2 * spawn_distance, road_h,
        facecolor="#555555", edgecolor="none", zorder=0
    ))

    # top sidewalk
    ax.add_patch(Rectangle(
        (-spawn_distance, road_y0 + road_h),
        2 * spawn_distance, curb_w,
        facecolor="#DDDDDD", edgecolor="none", zorder=0
    ))

    # -----------------------------
    # Road markings
    # -----------------------------
    ax.plot([-spawn_distance, spawn_distance], [road_y0, road_y0], color="white", lw=2, zorder=1)
    ax.plot([-spawn_distance, spawn_distance], [road_y0 + road_h, road_y0 + road_h], color="white", lw=2, zorder=1)

    ax.plot(
        [-spawn_distance, spawn_distance],
        [center_line_y, center_line_y],
        color="white",
        lw=2,
        linestyle=(0, (10, 10)),
        zorder=1
    )

    # -----------------------------
    # Crosswalk (zebra) at x=0
    # -----------------------------
    cw_half = 2.2
    ax.add_patch(Rectangle(
        (-cw_half, road_y0),
        2 * cw_half, road_h,
        facecolor="white", alpha=0.18,
        edgecolor="white", lw=2,
        zorder=2
    ))

    stripe_w, gap = 0.35, 0.22
    x = -cw_half
    i = 0
    while x < cw_half:
        if i % 2 == 0:
            ax.add_patch(Rectangle(
                (x, road_y0),
                stripe_w, road_h,
                facecolor="white", alpha=0.55,
                edgecolor="none",
                zorder=3
            ))
        x += stripe_w + gap
        i += 1

    ax.axvline(0, color="black", linestyle="--", lw=1.4, zorder=4)

    # -----------------------------
    # Cars
    # -----------------------------
    CAR_ZOOM = 0.13
    for x, _ in env_data.get("cars_lane1", []):
        draw_png(ax, IMG_CAR_LEFT, x, lane1_y, zoom=CAR_ZOOM, zorder=10)

    for x, _ in env_data.get("cars_lane2", []):
        draw_png(ax, IMG_CAR_RIGHT, x, lane2_y, zoom=CAR_ZOOM, zorder=10)

    # -----------------------------
    # Pedestrian (stage-based)
    # -----------------------------
    ped_stage = int(env_data.get("ped_stage", 0))
    stage_prog = float(env_data.get("stage_progress", 0.0))

    PED_ZOOM = 0.065

    if ped_stage == 0:
        # CURB waiting
        draw_png(ax, IMG_PED_WAIT, -8, bottom_sidewalk_y, zoom=PED_ZOOM, zorder=20)
        title = "WAITING at curb"
    elif ped_stage == 2:
        # MEDIAN waiting
        draw_png(ax, IMG_PED_WAIT, 0, median_y, zoom=PED_ZOOM, zorder=20)
        title = "WAITING at median"
    elif ped_stage == 1:
        # CROSSING lane1: move from curb -> median
        ped_y = bottom_sidewalk_y + stage_prog * (median_y - bottom_sidewalk_y)
        draw_png(ax, IMG_PED_WALK, 0, ped_y, zoom=PED_ZOOM, zorder=20)
        title = f"CROSSING lane 1 | progress={stage_prog:.2f}"
    elif ped_stage == 3:
        # CROSSING lane2: move from median -> top sidewalk
        ped_y = median_y + stage_prog * (top_sidewalk_y - median_y)
        draw_png(ax, IMG_PED_WALK, 0, ped_y, zoom=PED_ZOOM, zorder=20)
        title = f"CROSSING lane 2 | progress={stage_prog:.2f}"
    else:
        draw_png(ax, IMG_PED_WAIT, -8, bottom_sidewalk_y, zoom=PED_ZOOM, zorder=20)
        title = "WAITING"

    # Optional: mark terminal events visually if you pass it via env_data
    terminal_event = env_data.get("terminal_event", None)
    if terminal_event == "collision":
        title = "💥 COLLISION"
    elif terminal_event == "success":
        title = "✅ SUCCESS"
    elif terminal_event == "timeout":
        title = "⌛ TIMEOUT"

    if show_title:
        ax.set_title(title)

    # -----------------------------
    # Axes formatting
    # -----------------------------
    ax.set_xlim(-spawn_distance, spawn_distance)
    ax.set_ylim(road_y0 - curb_w - 0.4, road_y0 + road_h + curb_w + 0.4)
    ax.set_xlabel("x position (m), crossing line at x=0")
    ax.set_yticks([lane2_y, center_line_y, lane1_y])
    ax.set_yticklabels(["Lane 2 (→ to 0)", "Median", "Lane 1 (← to 0)"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    # Pause to control visual speed (match sim dt)
    plt.pause(float(dt))
