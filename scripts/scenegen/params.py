"""Scene parameters: bed layout, colours, crop scale and tool paths."""

import math

HARVEST_BOT_USD = (
    "/home/charlotte/"
    "harvestloop_sim/assets/harvest_bot.usd"
)

# =========================================================
# HARVEST TOOL GEOMETRY
# =========================================================
#
# GraspTip and CutterTip are rigid markers on the same tool.
# Their physical spacing is therefore fixed.
#
# Instead of generating GraspPoint and CutPoint independently,
# the crop is now generated so:
#
#     ||CutPoint - GraspPoint||
#         ==
#     ||CutterTip - GraspTip||
#
# The script measures this spacing directly from harvest_bot.usd
# after the robot reference is added to the stage.
GRASP_TIP_PATH = (
    "/World/HarvestBot/jackal/ur5e/"
    "Gripper/Robotiq_Hand_E_edit/base_link/GraspTip"
)

CUTTER_TIP_PATH = (
    "/World/HarvestBot/jackal/ur5e/"
    "Gripper/Robotiq_Hand_E_edit/base_link/CutterTip"
)

# Place the CutPoint slightly away from the main-stem attachment
# instead of directly at the junction.
CUTPOINT_OFFSET_FROM_PEDUNCLE_START = (
    0.015 * 0.60
)

# Keep some peduncle extending beyond the GraspPoint so the marker is
# not generated exactly on the distal end of the truss.
GRASPPOINT_DISTAL_MARGIN = (
    0.030 * 0.60
)

SEED = 42

# =========================================================
# FOLIAGE / OCCLUSION LEVEL
# =========================================================
#
# Experimental variable for fixed vs active perception.
# Leaves are the primary RGB-D occluders of peduncles. Stem /
# truss / fruit topology is drawn from the structure RNG; foliage
# uses a separate path-derived RNG so LOW/MEDIUM/HIGH keep the
# same truss paths and Jackal stops for a fixed SEED.
#
#   low    — sparse small leaves
#   medium — baseline (4–8 leaves, original size)
#   high   — denser / larger leaves

OCCLUSION_LEVEL = "medium"

OCCLUSION_LEVEL_CHOICES = (
    "low",
    "medium",
    "high",
)

# Alias used by harvestloop / paired runner.
OCCLUSION_CHOICES = OCCLUSION_LEVEL_CHOICES

OCCLUSION_LEAF_COUNT = {
    "low": (1, 3),
    "medium": (4, 8),
    "high": (9, 14),
}

OCCLUSION_LEAF_SIZE_SCALE = {
    "low": 0.70,
    "medium": 1.00,
    "high": 1.35,
}


def apply_scene_seed(seed):
    """Set the deterministic scene seed for the next build."""
    global SEED

    SEED = int(seed)
    return SEED


def apply_occlusion_level(level):
    """Set foliage occlusion for the next build."""
    global OCCLUSION_LEVEL

    level = str(level).strip().lower()

    if level not in OCCLUSION_LEVEL_CHOICES:
        raise ValueError(
            "Unknown occlusion level "
            f"{level!r}; expected one of "
            f"{OCCLUSION_LEVEL_CHOICES}"
        )

    OCCLUSION_LEVEL = level
    return OCCLUSION_LEVEL


def leaf_count_range():
    return OCCLUSION_LEAF_COUNT[OCCLUSION_LEVEL]


def leaf_size_scale():
    return OCCLUSION_LEAF_SIZE_SCALE[OCCLUSION_LEVEL]


ROW_COUNT = 4

# =========================================================
# CROP / GREENHOUSE SCALE
# =========================================================
#
# The UR5e in harvest_bot.usd is scaled to 0.6.
# Scale the crop morphology and crop-bed geometry accordingly.
#
# IMPORTANT:
# The Jackal itself is still full-size, so the driving aisle
# dimensions are NOT blindly multiplied by 0.6.
#

CROP_SCALE = 0.60


# =========================================================
# VISUAL APPEARANCE
# =========================================================
#
# Keep appearance deterministic for the first perception baseline.
# Do NOT randomize color or lighting yet.
#
# GraspPoint and CutPoint remain in USD as ground-truth markers for
# evaluation/control, but are hidden from rendered RGB by default.
SHOW_GROUND_TRUTH_MARKERS = False

# Fixed RGB colors in linear-ish [0, 1] preview values.
SCENE_COLORS = {
    "ground":       (0.62, 0.64, 0.66),   # light neutral gray
    "bed":          (0.24, 0.11, 0.045),  # dark soil brown
    "main_stem":    (0.16, 0.46, 0.12),   # medium green
    "peduncle":     (0.28, 0.58, 0.14),   # lighter green
    "branch":       (0.30, 0.62, 0.16),   # lighter green
    "leaf":         (0.055, 0.26, 0.065), # dark leaf green
    "tomato":       (0.78, 0.055, 0.035), # ripe tomato red
    "grasp_marker": (0.05, 0.35, 1.00),   # blue debug marker
    "cut_marker":   (1.00, 0.78, 0.02),   # yellow debug marker
}

SCENE_MATERIALS = {}


# Crop-bed geometry
BED_LENGTH = 2.5 * CROP_SCALE
BED_WIDTH = 0.35 * CROP_SCALE
BED_HEIGHT = 0.30 * CROP_SCALE

# Keep enough clearance for the full-size Jackal + basket.
CORRIDOR_WIDTH = 0.90
# Wider free aisle than the previous 0.55 m narrow experiment, still
# tighter than the baseline 0.80 m gap.
CROSS_AISLE_GAP = 0.65
FIRST_ROW_Y = 1.5

# =========================================================
# HARVESTABLE TRUSS HEIGHT
# =========================================================
#
# The UR5e in harvest_bot.usd is scaled to 0.6. Link lengths below
# are from assets/ur5e_scaled_06.urdf (already scaled). Aisle-1
# diagnosis: GraspPoint success collapses once the shoulder reach
# exceeds ~0.55 m; targets above that never had a chance and must
# not be authored into the scene.
#
# Mount: jackal base_link at z=0.065, ur5e xform translate z=0.333.

# From assets/ur5e_scaled_06.urdf (already at 0.6 scale):
#   upper arm 0.255 + forearm 0.23532 + grasp tip 0.042 ≈ 0.532 m
_UR5E_SHOULDER_ABOVE_BASE = 0.0975
_JACKAL_BASE_Z = 0.065
_UR5E_MOUNT_Z = 0.333

# Practical reach budget (stricter than the bare 0.532 m envelope).
MAX_HARVEST_SHOULDER_REACH = 0.55

# Half the row pitch: stem sits this far from the aisle centerline.
_AISLE_TO_STEM_Y = (
    BED_WIDTH
    + CROSS_AISLE_GAP
) / 2.0

_SHOULDER_WORLD_Z = (
    _JACKAL_BASE_Z
    + _UR5E_MOUNT_Z
    + _UR5E_SHOULDER_ABOVE_BASE
)

# Vertical room left after spending reach on the lateral standoff.
_MAX_GRASP_DZ = math.sqrt(
    max(
        0.0,
        MAX_HARVEST_SHOULDER_REACH ** 2
        - _AISLE_TO_STEM_Y ** 2,
    )
)

# Small allowance for peduncle pitch (±10 deg) and base drift.
_ATTACH_HEIGHT_MARGIN = 0.030

# attach_height is measured from the bed top (stem root). Grasp world
# Z ≈ BED_HEIGHT + attach_height.
MAX_HARVESTABLE_ATTACH_HEIGHT = (
    _SHOULDER_WORLD_Z
    + _MAX_GRASP_DZ
    - BED_HEIGHT
    - _ATTACH_HEIGHT_MARGIN
)
