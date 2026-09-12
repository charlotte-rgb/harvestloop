"""Tunable constants, USD paths and route geometry for the aisle-1 run."""

from pathlib import Path
import math

import numpy as np


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path("/home/charlotte/harvestloop_sim")
SCRIPTS_DIR = str(PROJECT_ROOT / "scripts")

# ============================================================
# WRIST CAMERA / VIEWPOINT CAPTURE
# ============================================================
#
# Permanent camera authored in harvest_bot.usd under gripper base_link.
# Referenced scene path:
WRIST_CAMERA_PATH = (
    "/World/HarvestBot/jackal/ur5e/"
    "Gripper/Robotiq_Hand_E_edit/base_link/WristCamera"
)

GRIPPER_BASE_PATH = (
    "/World/HarvestBot/jackal/ur5e/"
    "Gripper/Robotiq_Hand_E_edit/base_link"
)

# Documented local mounting (gripper base_link frame).
WRIST_CAMERA_LOCAL_POSITION = np.array(
    [0.075, -0.035, 0.085],
    dtype=np.float64,
)

WRIST_CAMERA_LOCAL_LOOK_TARGET = np.array(
    [0.0, 0.18, 0.05],
    dtype=np.float64,
)

VIEWPOINT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "perception_debug"
    / "aisle1_route_viewpoint"
)

# ============================================================
# RUN ARTIFACTS
# ============================================================
#
# Matched fixed/active pairs share one experiment folder:
#
#   results/runs/<experiment_id>/
#       fixed_view/
#       active_perception/
#
# <experiment_id> is usually YYYYMMDD_HHMMSS. Leave unset to start a
# new folder, or reuse the newest folder that still lacks this
# condition so the second Script Editor run lands beside the first.

RUN_ARTIFACTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "runs"
)

# Optional sticky id for the pair folder. None → auto (see
# resolve_run_experiment). Override:
#   EXPERIMENT = "20260909_192714"   # Script Editor
#   --experiment / HARVESTLOOP_EXPERIMENT
RUN_EXPERIMENT_ID = None

# Perception experiment protocol for this process.
#
# shared_view (default): one trial per truss. Capture canonical view
# once (= fixed-view result). If needs_another_view / view-1 fails,
# continue to views 2–3 in the SAME trial and score rescue.
# fixed_view: legacy one-capture-only policy (optional later e2e).
# active_perception: alias of shared_view (kept for old launchers).
#
# Override:
#   --condition shared_view|fixed_view|active_perception
#   HARVESTLOOP_CONDITION=...
RUN_CONDITION_LABEL = "shared_view"

RUN_CONDITION_CHOICES = (
    "shared_view",
    "fixed_view",
    "active_perception",
)

# Paired-experiment scene identity (logging + target lock).
# Does not change the controller. Applied by run_paired_experiment.
SCENE_SEED = 42
OCCLUSION_LEVEL = "medium"
OCCLUSION_CHOICES = (
    "low",
    "medium",
    "high",
)

# None = full aisle. Pilot/benchmark may cap attempted trusses.
MAX_PAIRED_TARGETS = None

# Legacy paired-protocol flags (off; shared-view does not use them).
PAIR_STATE_RESTORE = False
PAIR_VIEW1_SANITY = False

# ============================================================
# ACTIVE PERCEPTION (viewpoint acquisition only)
# ============================================================
#
# Does not change the estimator or the grasp/cut controller. Only
# decides whether to move the wrist camera and capture again.

ACTIVE_PERCEPTION_MAX_VIEWS = 3

# Lateral camera offsets along world +X / -X from the canonical
# viewpoint, still aimed at the same look target.
ACTIVE_VIEW_LATERAL_OFFSET_M = 0.080

# Ordered viewpoint names. Index 0 is always the fixed-view canonical.
ACTIVE_VIEWPOINT_SEQUENCE = (
    "canonical",
    "lateral_plus_x",
    "lateral_minus_x",
)

# Perception seam behind estimate_target(...). Controllers consume
# only that output.
#   oracle_usd     — returns hidden USD GT (wiring check)
#   rgbd_peduncle  — segmentation-first peduncle extent (superseded)
#   keypoints      — two task keypoints per truss (baseline)
ESTIMATOR_MODE = "keypoints"

# ============================================================
# KEYPOINT PERCEPTION
# ============================================================
#
# K1 = CutPoint, K2 = GraspPoint. After lift:
#   CutPoint = K1, GraspPoint = K2
#   peduncle_direction = normalize(K2 - K1)
#
# Placement on the detected peduncle matches scenegen:
#   K1 = stem_end + dir * cut_offset
#   K2 = K1       + dir * tool_spacing

# How far cut sits from the peduncle/stem junction (CROP_SCALE=0.6).
KEYPOINT_CUT_OFFSET_M = 0.009

# ||K2 - K1|| should match tool tip spacing; wider than that means the
# two keypoints did not land on the cut/grasp pair.
KEYPOINT_SPACING_TOL_M = 0.012
KEYPOINT_BAD_SPACING_PENALTY = 0.35

# Authored peduncle length range, for the supporting segment only.
KEYPOINT_LENGTH_MIN_M = 0.080
KEYPOINT_LENGTH_MAX_M = 0.200
KEYPOINT_LENGTH_TARGET_M = 0.130

# Reject nearly-vertical bars (main stem / leaf plate). Peduncles in
# this scene leave the stem roughly horizontally toward the aisle.
KEYPOINT_MAX_VERTICAL_DOT = 0.55

# Fruit cluster must be a real tomato group, not a specular speck.
KEYPOINT_MIN_FRUIT_PIXELS = 80
KEYPOINT_FRUIT_CLUSTER_RADIUS_M = 0.08
# Try the best few fruit blobs; the aimed one is usually near centre,
# but a larger neighbour can otherwise steal the seed.
KEYPOINT_FRUIT_CANDIDATES = 3

# Stem end of a real peduncle sits this far from the fruit; shorter
# means we latched onto a tomato-side stub.
KEYPOINT_MIN_STEM_TO_FRUIT_M = 0.070

# Depth gating for wrist captures of an adjacent truss.
KEYPOINT_DEPTH_MIN = 0.12
KEYPOINT_DEPTH_MAX = 0.55
KEYPOINT_DEPTH_BAND_M = 0.06
KEYPOINT_DEPTH_SPECKLE_M = 0.07

KEYPOINT_MIN_TOMATO_PIXELS = 40
KEYPOINT_MIN_PEDUNCLE_PIXELS = 30

# Only used to decide WHICH green blob belongs to this fruit; the
# whole blob is then kept, so the peduncle is not clipped to a stub.
KEYPOINT_TOMATO_DILATION = 12

# No peduncle pixel can be further from its own fruit than a peduncle
# is long. Keeps the blob from walking down the main stem.
KEYPOINT_MAX_REACH_M = 0.22

# Crossing peduncles land in one blob but are separated by a gap along
# the fitted axis. Only the run touching this truss's fruit is kept.
KEYPOINT_SEGMENT_GAP_M = 0.02

# Robust endpoints: the 5th / 95th percentile along the peduncle
# axis, so a single stray pixel cannot define a keypoint.
KEYPOINT_EXTREME_PERCENTILE = 5.0

KEYPOINT_REFIT_RESIDUAL_M = 0.015
KEYPOINT_INLIER_OFFSET_M = 0.010

# Picking this truss's peduncle out of the crossing ones: sample
# candidate lines, keep the best-supported segment inside a thin tube
# that both reaches the fruit and has a plausible peduncle length.
KEYPOINT_TUBE_RADIUS_M = 0.015
KEYPOINT_TRACK_MAX_LENGTH_M = 0.190
KEYPOINT_ATTACHMENT_MAX_M = 0.100
KEYPOINT_RANSAC_ITERATIONS = 600
KEYPOINT_RANSAC_MAX_POINTS = 2000

# Per-keypoint confidence inputs.
KEYPOINT_DEPTH_PATCH_RADIUS = 3
KEYPOINT_SUPPORT_WINDOW_M = 0.015
KEYPOINT_SUPPORT_TARGET_PIXELS = 25
KEYPOINT_BORDER_MARGIN_PX = 40.0
KEYPOINT_LINE_OFFSET_BAD_M = 0.020

# Below these, the estimate is flagged `needs_another_view`. The
# fixed-view condition only records the flag; the active-perception
# condition will use it to move the wrist camera and re-estimate.
KEYPOINT_MIN_CONFIDENCE = 0.45
KEYPOINT_MIN_SINGLE_CONFIDENCE = 0.30

# Write each overlay as soon as that truss is estimated, and show it
# in an Isaac window so detection can be watched during the route.
SHOW_KEYPOINT_OVERLAY = True

WRIST_CAPTURE_RESOLUTION = (
    640,
    480,
)

WRIST_CAPTURE_WARMUP_FRAMES = 3
WRIST_CAPTURE_RENDER_STEPS = 1

# ============================================================
# RUN SPEED
# ============================================================
#
# Neither switch changes what the route computes: same control
# steps, same physics dt, same tolerances. They remove real time
# spent waiting and rendering between those steps.

# The editor app caps its main loop at 60 Hz, so every control step
# waits for the next tick even when the step finished long before
# it. A route is tens of thousands of steps, and that wait is most
# of the wall clock. Disabled for the run, restored afterwards (the
# editor UI is sluggish while it is off).
UNTHROTTLE_RUN_LOOP = True

# The wrist render product renders on every app update once it is
# created, for the whole route, to produce two saved images per
# truss. Switch it on only around a capture.
GATE_WRIST_RENDER_PRODUCT = True

# The main viewport also renders every update, and with the throttle
# off that render is what each step waits on. Freezing it is the
# largest remaining saving and still changes nothing about the run —
# the wrist captures come from their own render product.
FREEZE_VIEWPORT_DURING_ROUTE = False

# Aisle-side GraspTip standoff from the stem axis (Y only).
# Per stem, viewpoint XY is fixed; only Z changes along the stem.
# 12 cm was too tight for stable RMPflow reach from the aisle park;
# tip settled ~20 cm from the stem and oscillated above tolerance.
VIEW_OFFSET = 0.200

VIEW_TOLERANCE = 0.050
# If the hard tolerance is never hit, still accept a stable near-miss
# so one unreachable truss does not abort the whole route test.
VIEW_ACCEPT_TOLERANCE = 0.090
VIEW_MAX_STEPS = 280
VIEW_SETTLE_FRAMES = 4
VIEW_PRINT_EVERY = 40
VIEW_NO_IMPROVE_STEPS = 40

# ============================================================
# AIMED VIEWPOINT
# ============================================================
#
# The viewpoint used to be a GraspTip position with no orientation
# target, so RMPflow was free to leave the wrist pointing anywhere
# and the truss landed outside the image in 23 of 26 captures. The
# viewpoint is now specified as a CAMERA pose: stand off the stem on
# the aisle side and look at the truss.

# Camera distance from the truss along the aisle normal.
#
# The two stem rows sit about 0.75 m apart, so the aisle half-width
# is ~0.375 m and the arm base stands on the centerline. A standoff
# near 0.35 m would park the camera on the arm's own vertical axis,
# in the shoulder singularity. 0.25 m keeps it ~0.12 m out toward
# the bed, and the ~60 deg horizontal field of view still frames
# ~0.29 m of stem, several times the width of a truss.
VIEW_CAMERA_STANDOFF = 0.250

# The truss must land inside this fraction of the frame, measured
# from the image center. 0.5 keeps it within the central half.
VIEW_AIM_CENTER_FRACTION = 0.50

# Outer command-measure-correct attempts. The first attempt commands
# the pose derived from the measured camera mount; each further one
# corrects for whatever the arm actually achieved.
VIEW_AIM_MAX_ATTEMPTS = 3

# If the aimed pose cannot be reached at all, fall back to the old
# position-only viewpoint so a hard-to-reach truss still gets an
# image and the route behaves as it did before.
VIEW_AIM_FALLBACK_TO_POSITION_ONLY = True

# Pair left/right stems into one stop when |dx| is within this.
STEM_PAIR_X_TOLERANCE = 0.28

# Park BEFORE each stem workspace when driving toward -X.
# Facing -X, "ahead" is more negative X, so the Jackal stop must sit
# at a larger (less negative / more positive) X than the stem.
#
# IMPORTANT:
# Do NOT clamp stops to the crop-bed X span. That previously forced
# stop_x ~= LEFT_BED_INNER_X (~-0.53) and erased large offsets, so the
# Jackal parked on/past the stems.
STOP_BEFORE_STEM_OFFSET = 0.40

# Soft driveline margins when clamping aisle parks.
STOP_X_MARGIN = 0.08

# If the Jackal is already this close to (or past, toward -X) the next
# stem stop, do not replan a drive — consecutive left/right stems often
# share nearly the same stop_x and turning around looks "weird".
AISLE_STOP_SKIP_TOLERANCE = 0.10

# Coarser aisle waypoints = fewer stop checks while driving.
AISLE_WAYPOINT_STEP = 0.45


# ============================================================
# USD PATH
# ============================================================

JACKAL_BASE_PATH = "/World/HarvestBot/jackal/base_link"

ARM_PATH = "/World/HarvestBot/jackal/ur5e"
ARM_BASE_PATH = "/World/HarvestBot/jackal/ur5e/base_link"

GRASP_TIP_PATH = (
    "/World/HarvestBot/jackal/ur5e/"
    "Gripper/Robotiq_Hand_E_edit/base_link/GraspTip"
)

CUTTER_TIP_PATH = (
    "/World/HarvestBot/jackal/ur5e/"
    "Gripper/Robotiq_Hand_E_edit/base_link/CutterTip"
)

# Aisle 1 beds flanking y=2.05 on the LEFT side of the central corridor.
# Facing -X: LEFT = +Y = north = Row_02_Left
#            RIGHT = -Y = south = Row_01_Left
AISLE1_LEFT_ROW_ROOT = "/World/Greenhouse/Row_02_Left"
AISLE1_RIGHT_ROW_ROOT = "/World/Greenhouse/Row_01_Left"

# Kept for shared helpers that still reference TARGET_ROW_ROOT.
TARGET_ROW_ROOT = AISLE1_LEFT_ROW_ROOT

AISLE1_SIDE_CONFIGS = (
    {
        "side_name": "left",
        "row_root": AISLE1_LEFT_ROW_ROOT,
        "approach_offset": np.array(
            [0.0, -VIEW_OFFSET, 0.0],
            dtype=np.float64,
        ),
    },
    {
        "side_name": "right",
        "row_root": AISLE1_RIGHT_ROW_ROOT,
        "approach_offset": np.array(
            [0.0, +VIEW_OFFSET, 0.0],
            dtype=np.float64,
        ),
    },
)

# Basket components created in harvest_bot.usd.
# DropPoint is deliberately NOT an obstacle.
BASKET_COMPONENT_NAMES = (
    "Bottom",
    "Front",
    "Rear",
    "Left",
    "Right",
)

ACTIVE_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Manually captured UR5e pose with the end effector positioned
# above the basket / DropPoint.
#
# IMPORTANT:
# This order is NOT the articulation's raw DOF order.
# It matches ACTIVE_ARM_JOINTS:
#   shoulder_pan_joint
#   shoulder_lift_joint
#   elbow_joint
#   wrist_1_joint
#   wrist_2_joint
#   wrist_3_joint
UR5E_HOME_Q = np.array(
    [
        +0.000000,  # shoulder_pan_joint
        -1.244506,  # shoulder_lift_joint
        -2.040756,  # elbow_joint
        -2.159941,  # wrist_1_joint
        +2.264363,  # wrist_2_joint
        -0.000257,  # wrist_3_joint
    ],
    dtype=np.float64,
)

# Known-safe upright pose.
# Used only while Jackal is parked, before and after RMPflow.
UR5E_UPRIGHT_Q = np.array(
    [
        0.0,             # shoulder_pan_joint
        -math.pi / 2.0,  # shoulder_lift_joint
        0.0,             # elbow_joint
        0.0,             # wrist_1_joint
        0.0,             # wrist_2_joint
        0.0,             # wrist_3_joint
    ],
    dtype=np.float64,
)

ARM_HOME_TOLERANCE = 0.055

# If the arm plateaus just outside the hard home tol (common residual
# after grasp/cut/release), accept once settled rather than aborting.
ARM_HOME_ACCEPT_TOLERANCE = 0.120
ARM_HOME_PLATEAU_STEPS = 30

# ------------------------------------------------------------
# RMPFLOW C-SPACE BASKET TRANSITION
# ------------------------------------------------------------
#
# Success = joints within tolerance of UR5E_HOME_Q (basket transport /
# release pose). DropPoint geometry is diagnostic only.
RMPFLOW_BASKET_CSPACE_TOLERANCE = 0.040
RMPFLOW_BASKET_CSPACE_MAX_STEPS = 140
RMPFLOW_BASKET_CSPACE_PRINT_EVERY = 40
RMPFLOW_BASKET_CSPACE_SETTLE_FRAMES = 4

# Fail only after sustained clear divergence from the best observed
# c-space error. Do not fail on one noisy frame.
RMPFLOW_BASKET_CSPACE_DIVERGENCE_MARGIN = 0.12
RMPFLOW_BASKET_CSPACE_DIVERGENCE_STEPS = 12

# ------------------------------------------------------------
# BASKET TASK-SPACE DIAGNOSTIC REGION
# ------------------------------------------------------------
#
# Recorded at route setup while the arm is at UR5E_HOME_Q:
#
#     GraspTip - DropPoint
#
# in the Basket-local frame. Logged during basket transport; release
# success is joint-home matching, not this region.
BASKET_DROP_XY_TOLERANCE = 0.050  # 5 cm around known-good XY relation
BASKET_DROP_Z_TOLERANCE = 0.090   # was 6 cm; park/payload often leaves ~8 cm Z
BASKET_DROP_3D_TOLERANCE = 0.095

# Soft accept bands retained for DropPoint diagnostics only.
BASKET_DROP_ACCEPT_XY_TOLERANCE = 0.060
BASKET_DROP_ACCEPT_Z_TOLERANCE = 0.100
BASKET_DROP_ACCEPT_3D_TOLERANCE = 0.110

# After entering the drop region, freeze the ACTUAL q and verify that
# the task-space relation remains valid before enabling basket physics.


# ------------------------------------------------------------
# PHYSICAL BASKET RELEASE
# ------------------------------------------------------------
#
# Truss roots are prepared as kinematic rigid bodies BEFORE physics.
# During carry they remain kinematic and their collisions are disabled.
#
# At basket release:
#   - stop logical following
#   - restore original truss collisions
#   - zero rigid-body linear/angular velocity
#   - switch kinematicEnabled -> False
#   - gravity then drops the truss into the physical basket
#
# Keep Jackal parked and hold the arm at basket pose while the truss
# falls/settles.
PHYSICAL_DROP_SETTLE_FRAMES = 30
PHYSICAL_DROP_PRINT_EVERY = 30
# Frames after enabling basket colliders before making the truss dynamic.
PHYSICAL_DROP_COLLISION_WARMUP_FRAMES = 6

# ------------------------------------------------------------
# BASKET COLLISION + CCD
# ------------------------------------------------------------
#
# RMPflow obstacle wrappers are NOT PhysX colliders. Before physics
# starts, explicitly apply CollisionAPI to physical geometry for all
# five basket components. Collision remains disabled until release.
#
# The harvested truss gets rigid-body CCD so the small falling payload
# does not tunnel through the thin basket bottom between physics steps.
BASKET_CONTACT_OFFSET = 0.025
BASKET_REST_OFFSET = 0.001

# ------------------------------------------------------------
# INVISIBLE THICK BASKET CATCH FLOOR
# ------------------------------------------------------------
#
# The visible Bottom is thin. Even with CCD, a small harvested truss can
# tunnel through a thin contact surface. We therefore pre-create a thick,
# invisible collider directly under the Basket Bottom.
#
# It is parented under the Basket, so it moves with the Jackal.
# Its PhysX collision remains OFF during all robot/arm motion and is
# enabled only immediately before the harvested truss becomes dynamic.
BASKET_CATCH_FLOOR_NAME = "PhysicsCatchFloor"

# Low invisible retaining walls that keep released trusses in the
# basket while the visible components are softened for the arm
# approach. Heights are clamped to the visible rim minus the clearance
# below, so the walls can never fight the gripper.
BASKET_CATCH_WALL_NAMES = (
    "PhysicsCatchWallXMin",
    "PhysicsCatchWallXMax",
    "PhysicsCatchWallYMin",
    "PhysicsCatchWallYMax",
)

BASKET_CATCH_WALL_HEIGHT = 0.160          # above the catcher floor top
BASKET_CATCH_WALL_THICKNESS = 0.040       # grows outward, not inward
BASKET_CATCH_WALL_INNER_INSET = 0.010     # inner face inside the footprint
BASKET_CATCH_WALL_RIM_CLEARANCE = 0.020   # stay this far below the rim
BASKET_CATCH_FLOOR_THICKNESS = 0.180   # 18 cm thick catcher (anti-tunnel)
BASKET_CATCH_FLOOR_XY_MARGIN = 0.040   # 4 cm extra around Bottom
BASKET_CATCH_FLOOR_TOP_OVERLAP = 0.008 # top sits 8 mm above Bottom top

ENABLE_TRUSS_CCD = True
ENABLE_TRUSS_SPECULATIVE_CCD = True

ARM_HOME_MAX_STEPS = 180

# ------------------------------------------------------------
# Deterministic pregrasp test
# ------------------------------------------------------------

LEFT_MANIP_X = -1.00

PREGRASP_OFFSET = 0.080
PREGRASP_TOLERANCE = 0.030
PREGRASP_MAX_STEPS = 350

# Final approach test:
# PREGRASP -> actual GraspPoint -> PREGRASP
#
# The previous run physically reached 0.0229 m from the GraspPoint,
# then RMPflow wandered away because the old threshold was 0.020 m.
# Treat <= 2.5 cm as immediate success, and also remember the best
# physical pose so a near-target solution is never lost.
# Logical grasp stage: accept immediately once the PHYSICAL GraspTip
# is within 3 cm of GraspPoint. Do not let RMPflow move away and then
# try to reconstruct an old "best" pose.
GRASP_TOLERANCE = 0.030
GRASP_TOLERANCE_EPS = 0.0005

# ------------------------------------------------------------
# BOUNDED GRASP RETRIES
# ------------------------------------------------------------
#
# Do not run one long RMPflow approach and hope it eventually converges.
# RMPflow is a local reactive policy; if an approach plateaus, stop it,
# reset to a nearby PREGRASP seed, and retry from a slightly different
# local branch.
GRASP_MAX_ATTEMPTS = 3
GRASP_ATTEMPT_MAX_STEPS = 100
GRASP_PROGRESS_PRINT_EVERY = 25

# Plateau = no meaningful reduction of the best Cartesian error.
GRASP_PLATEAU_MIN_IMPROVEMENT = 0.0015  # 1.5 mm
GRASP_PLATEAU_CONSECUTIVE_STEPS = 25

# Stop an attempt if it clearly walks away from its best pose.
GRASP_DIVERGENCE_MARGIN = 0.015
GRASP_DIVERGENCE_CONSECUTIVE_STEPS = 10

# Before retry 2/3, move back to a nearby aisle-side PREGRASP seed.
GRASP_RETRY_RESET_TOLERANCE = 0.035
GRASP_RETRY_RESET_MAX_STEPS = 120
GRASP_RETRY_RESET_PRINT_EVERY = 30

# Slightly different Z seeds change the local approach branch without
# changing the actual GraspPoint target.
GRASP_RETRY_PREGRASP_Z_OFFSETS = (
    0.000,
    +0.012,
    -0.012,
)

# Replay each successful approach path backward instead of asking
# RMPflow to solve an independent retreat problem.
#
# IMPORTANT:
# The old retreat gave each recorded waypoint only 2 frames. After CUT
# the arm showed ~0.15-0.16 rad tracking lag and never recovered the
# PREGRASP pose. Now every reverse waypoint is held until the physical
# joints catch up (or a bounded timeout is reached).
GRASP_RETURN_WAYPOINT_TOLERANCE = 0.030
GRASP_RETURN_WAYPOINT_MAX_FRAMES = 12
# Joint tolerance is now diagnostic only at PREGRASP recovery.
GRASP_RETURN_FINAL_JOINT_TOLERANCE = 0.035

# Physical task-space criterion for successful post-CUT retreat.
GRASP_RETURN_FINAL_CART_TOLERANCE = 0.040
GRASP_RETURN_FINAL_HOLD_MAX_FRAMES = 60

# ------------------------------------------------------------
# POST-CUT RECEDING-HORIZON CARTESIAN RETREAT
# ------------------------------------------------------------
#
# Previous fixed staged waypoints could be "accepted" without the arm
# actually moving, which made later fixed waypoints jump too far away.
#
# New controller:
#   while GraspTip is outside the final PREGRASP region:
#       1. read ACTUAL physical GraspTip
#       2. compute direction toward PREGRASP
#       3. create a short target 1 cm ahead from the CURRENT tip
#       4. let RMPflow move only briefly toward that target
#       5. re-read ACTUAL tip and regenerate the next target
#
# This is a receding-horizon / moving-target retreat. It never relies
# on skipped historical waypoints and never restores old joint poses.
POSTCUT_RETREAT_TOLERANCE = 0.040

# Short Cartesian step generated from the ACTUAL current physical tip.
POSTCUT_RETREAT_STEP_DISTANCE = 0.020  # 2.0 cm

# IMPORTANT:
# This MUST be substantially smaller than POSTCUT_RETREAT_STEP_DISTANCE.
# Otherwise a freshly generated target is considered "already
# reached" before the arm has actually moved.
POSTCUT_RETREAT_LOCAL_TARGET_TOLERANCE = 0.006  # 6 mm

# Give the local policy enough frames to physically approach the short
# receding target before we re-measure and regenerate.
POSTCUT_RETREAT_FRAMES_PER_TARGET = 8

# Total number of receding targets allowed before declaring failure.
POSTCUT_RETREAT_MAX_TARGETS = 40

POSTCUT_RETREAT_PRINT_EVERY_TARGET = 2

# Require real progress toward PREGRASP over several consecutive
# receding targets. Tiny numerical noise is ignored.
POSTCUT_RETREAT_MIN_PROGRESS = 0.0015
POSTCUT_RETREAT_NO_PROGRESS_LIMIT = 6

# If PREGRASP error grows this much above the best seen error for many
# receding targets, stop instead of letting RMPflow run away.
POSTCUT_RETREAT_DIVERGENCE_MARGIN = 0.020
POSTCUT_RETREAT_DIVERGENCE_LIMIT = 4

# Bounded retreat attempts, like the CUT stage.
#
# IMPORTANT:
# Retries are recovery only. The success criterion stays exactly
# POSTCUT_RETREAT_TOLERANCE so fixed-view and active-perception runs
# are scored by the same controller. A diverging attempt is stopped,
# the best observed retreat state is restored and remeasured, and the
# receding-horizon retreat runs again from there.
POSTCUT_RETREAT_MAX_ATTEMPTS = 3
POSTCUT_RETREAT_BEST_RESTORE_HOLD_FRAMES = 12

# A failed retreat leaves an already-severed truss on the tool, and
# going straight to the basket would bypass the retreat controller.
# The truss is therefore released where it is, the arm recovers SAFE
# UPRIGHT and the route continues with the failure logged under
# RETREAT. Set False to abort the whole run on the first such failure.
POSTCUT_RETREAT_FAILURE_CONTINUES_ROUTE = True

# ------------------------------------------------------------
# Grasp -> cutter alignment -> CUT
# ------------------------------------------------------------
#
# GraspTip and CutterTip are fixed markers on the same rigid tool.
# At the cut stage we keep GraspTip at GraspPoint and rotate the tool
# so the vector GraspTip->CutterTip aligns with
# GraspPoint->CutPoint.
# Logical CUT acceptance region.
#
# Accept the first physical pose satisfying:
#   GraspTip  -> GraspPoint <= 3 cm
#   CutterTip -> CutPoint   <= 4 cm
#
# Once this region is entered, CUT is declared immediately.
#
# Hard tolerances are FIXED for perception experiments. Robustness comes
# from bounded retries + best-pose restore/remeasure, NOT soft accept.
CUT_GRASP_HOLD_TOLERANCE = 0.030
CUTTER_TOLERANCE = 0.040

# Numerical allowance for floating-point boundary cases.
CUT_TOLERANCE_EPS = 0.001

# Bounded CUT attempts (same spirit as GRASP retries).
CUT_MAX_ATTEMPTS = 3
CUT_ATTEMPT_MAX_STEPS = 140
CUT_PROGRESS_PRINT_EVERY = 25

# Divergence watchdog uses a near-score only to detect drift away from
# the best pose. It is NEVER an automatic CUT success condition.
CUT_NEAR_SCORE = 1.25
CUT_DIVERGENCE_MARGIN = 0.20
CUT_DIVERGENCE_CONSECUTIVE_STEPS = 8

# After diverge/timeout: restore best_q, hold, then remeasure against
# the original 3 cm / 4 cm criterion.
CUT_BEST_RESTORE_HOLD_FRAMES = 30

# Before CUT retries 2/3, back off toward a nearby aisle-side seed
# (nominal PREGRASP + small Z), then re-approach GraspPoint.
CUT_RETRY_SEED_Z_OFFSETS = (
    0.000,
    +0.012,
    -0.012,
)
CUT_RETRY_RESET_TOLERANCE = 0.035
CUT_RETRY_RESET_MAX_STEPS = 120
CUT_RETRY_REGRASP_TOLERANCE = 0.030
CUT_RETRY_REGRASP_MAX_STEPS = 100

# Replay the successful GRASP->CUT trajectory backward after CUT.
CUT_RETURN_WAYPOINT_TOLERANCE = 0.035
CUT_RETURN_WAYPOINT_MAX_FRAMES = 10

# The harvested truss remains the same USD prim. We prepare a
# translation XformOp before physics initialization. After CUT,
# that op follows the GraspTip displacement. No prim is created,
# removed, or reparented while PhysX tensor views are active.
TRUSS_CARRY_OP_SUFFIX = "harvestCarry"

# Replay the successful PREGRASP joint trajectory backward instead
# of asking RMPflow to solve a separate return-to-upright problem.
#
# IMPORTANT:
# Do not advance to the next recorded joint configuration until the
# physical arm has substantially caught up. The old fixed 2-frame
# replay kept ~0.06-0.07 rad of tracking lag all the way back.
RETURN_REPLAY_STRIDE = 6
RETURN_WAYPOINT_TOLERANCE = 0.050
RETURN_WAYPOINT_MAX_FRAMES = 4

# Exact historical joint reproduction is diagnostic only.
RETURN_FINAL_TOLERANCE = 0.045

# SAFE UPRIGHT validation:
#
# 1. exact/near joint return:
#       max |q - recorded_upright_q| <= RETURN_FINAL_TOLERANCE
#
# OR
#
# 2. GraspTip returns to the same pose REGION RELATIVE TO UR5e base_link:
#       local-tip error <= RETURN_UPRIGHT_CART_TOLERANCE
#
# Do NOT use world-space GraspTip displacement as the pass/fail
# criterion because small Jackal drift moves the whole UR5e in world.
RETURN_UPRIGHT_CART_TOLERANCE = 0.050

RETURN_FINAL_HOLD_MAX_FRAMES = 20

TARGET_MAX_BASE_DISTANCE = 0.75

# Target must lie AHEAD of the parked Jackal, not behind it.
# At the first aisle the Jackal faces approximately -X.
TARGET_MIN_FORWARD_DISTANCE = 0.08
TARGET_MAX_FORWARD_DISTANCE = 0.60

RMPFLOW_PHYSICS_DT = 1.0 / 60.0
RMPFLOW_MAX_SUBSTEP = 0.00334

URDF_FILENAME = "ur5e_scaled_06.urdf"
ROBOT_DESCRIPTION_FILENAME = "ur5e_robot_description_scaled_06.yaml"
RMPFLOW_CONFIG_FILENAME = "ur5e_rmpflow_config_scaled_06.yaml"


# ============================================================
# EXACT GREENHOUSE GEOMETRY
# Matches build_scene_colored_narrow_aisle.py
# ============================================================

CROP_SCALE = 0.60

BED_LENGTH = 2.5 * CROP_SCALE       # 1.50 m
BED_WIDTH = 0.35 * CROP_SCALE       # 0.21 m (narrower beds)
CORRIDOR_WIDTH = 0.90
# Matches build_scene_colored_narrow_aisle.py
CROSS_AISLE_GAP = 0.65
FIRST_ROW_Y = 1.50

ROW_PITCH = (
    BED_WIDTH
    + CROSS_AISLE_GAP
)

ROW_1_Y = FIRST_ROW_Y
ROW_2_Y = FIRST_ROW_Y + ROW_PITCH

# First cross aisle lies between Row 1 and Row 2.
AISLE_LOWER_Y = (
    ROW_1_Y
    + BED_WIDTH / 2.0
)

AISLE_UPPER_Y = (
    ROW_2_Y
    - BED_WIDTH / 2.0
)

# Exact center of first cross aisle.
AISLE_Y = (
    AISLE_LOWER_Y
    + AISLE_UPPER_Y
) / 2.0

# Exact center of central corridor.
CORRIDOR_X = 0.00

# Bed placement from build_scene:
SIDE_OFFSET = (
    CORRIDOR_WIDTH / 2.0
    + BED_LENGTH / 2.0
)

LEFT_BED_CENTER_X = -SIDE_OFFSET
RIGHT_BED_CENTER_X = +SIDE_OFFSET

LEFT_BED_OUTER_X = (
    LEFT_BED_CENTER_X
    - BED_LENGTH / 2.0
)

LEFT_BED_INNER_X = (
    LEFT_BED_CENTER_X
    + BED_LENGTH / 2.0
)

RIGHT_BED_INNER_X = (
    RIGHT_BED_CENTER_X
    - BED_LENGTH / 2.0
)

RIGHT_BED_OUTER_X = (
    RIGHT_BED_CENTER_X
    + BED_LENGTH / 2.0
)

# Deterministic manipulation parking point in the first aisle.
LEFT_END_X = LEFT_MANIP_X

# Explicit Jackal start pose.
HOME_X = 0.00
HOME_Y = 0.00
HOME_YAW = 0.00

CORRIDOR_YAW = math.pi / 2.0
LEFT_YAW = math.pi

START_XY_TOLERANCE = 0.002
START_YAW_TOLERANCE = math.radians(0.5)
START_RESET_ATTEMPTS = 6
START_RESET_SETTLE_FRAMES = 8


# ============================================================
# DENSE WAYPOINTS
# ============================================================

# Central corridor: exact x = 0.
CENTRAL_FORWARD_WAYPOINTS = [
    (CORRIDOR_X, 0.30),
    (CORRIDOR_X, 0.60),
    (CORRIDOR_X, 0.90),
    (CORRIDOR_X, 1.20),
    (CORRIDOR_X, 1.50),
    (CORRIDOR_X, 1.80),
    (CORRIDOR_X, AISLE_Y),
]

# First cross aisle: exact y = 2.05.
LEFT_FORWARD_WAYPOINTS = [
    (-0.25, AISLE_Y),
    (-0.50, AISLE_Y),
    (-0.75, AISLE_Y),
    (LEFT_MANIP_X, AISLE_Y),
]

# Reverse through the same path.
LEFT_REVERSE_WAYPOINTS = [
    (-0.75, AISLE_Y),
    (-0.50, AISLE_Y),
    (-0.25, AISLE_Y),
    (CORRIDOR_X, AISLE_Y),
]

CENTRAL_REVERSE_WAYPOINTS = [
    (CORRIDOR_X, 1.80),
    (CORRIDOR_X, 1.50),
    (CORRIDOR_X, 1.20),
    (CORRIDOR_X, 0.90),
    (CORRIDOR_X, 0.60),
    (CORRIDOR_X, 0.30),
    (HOME_X, HOME_Y),
]


# ============================================================
# JACKAL GEOMETRY
# ============================================================

WHEEL_RADIUS = 0.098
TRACK_WIDTH = 0.375

LEFT_WHEELS = [
    "front_left_wheel_joint",
    "rear_left_wheel_joint",
]

RIGHT_WHEELS = [
    "front_right_wheel_joint",
    "rear_right_wheel_joint",
]


# ============================================================
# SPEEDS
# ============================================================

MAX_MAIN_SPEED = 1.80
MAX_AISLE_SPEED = 1.50

# Direct point-following is allowed to steer more strongly than
# the previous tiny lateral correction controller.
MAX_DRIVE_ANGULAR_SPEED = 1.30

# Fast in-place turns.
MAX_TURN_ANGULAR_SPEED = 3.20
MIN_TURN_ANGULAR_SPEED = 1.10

# Strong heading errors automatically reduce linear speed.
HEADING_SLOW_ANGLE = math.radians(12.0)
HEADING_VERY_SLOW_ANGLE = math.radians(28.0)

HEADING_SLOW_SPEED = 0.95
HEADING_VERY_SLOW_SPEED = 0.60


# ============================================================
# CONTROLLER GAINS
# ============================================================

K_LINEAR = 1.55

K_POINT_YAW = 2.40
K_TURN_YAW = 4.00

# Pure-pursuit style lookahead on the EXACT greenhouse centerline.
# This prevents each dense waypoint from creating a new diagonal line
# from the robot's current drifted position.
CENTERLINE_LOOKAHEAD = 0.65

# If lateral drift grows, reduce speed while keeping steering active.
# Tight thresholds keep the Jackal clear of the beds on aisle-1.
CENTERLINE_DRIFT_SLOW = 0.05
CENTERLINE_DRIFT_VERY_SLOW = 0.10
CENTERLINE_DRIFT_SPEED = 0.70
CENTERLINE_DRIFT_VERY_SLOW_SPEED = 0.40


# ============================================================
# ACCELERATION LIMITS
# ============================================================

CONTROL_DT = 1.0 / 60.0

MAX_LINEAR_ACCEL = 2.40
MAX_LINEAR_DECEL = 3.00

DRIVE_ANGULAR_ACCEL = 2.40
DRIVE_ANGULAR_DECEL = 3.00

TURN_ANGULAR_ACCEL = 6.00
TURN_ANGULAR_DECEL = 7.00


# ============================================================
# TOLERANCES
# ============================================================

PASS_THROUGH_TOLERANCE = 0.10
FINAL_WAYPOINT_TOLERANCE = 0.040

# IMPORTANT:
# Waypoint completion is evaluated mainly along the direction of travel.
# This prevents lateral error from making the Jackal drive past an
# intersection that it has already reached.
PASS_LONGITUDINAL_TOLERANCE = 0.08
FINAL_LONGITUDINAL_TOLERANCE = 0.030

TURN_TOLERANCE = math.radians(5.0)

MAX_SEGMENT_STEPS = 2500
MAX_TURN_STEPS = 3000

PRINT_INTERVAL = 35


# ============================================================
# TIP SAFETY
# ============================================================

TIP_WARNING = math.radians(7.0)
TIP_HARD_STOP = math.radians(12.0)


def apply_run_condition(condition):
    """
    Set RUN_CONDITION_LABEL for this process.

    Call before RunArtifacts() so the run directory name matches.
    Also refreshes star-imported copies already loaded in harvest /
    runlog / route / metrics.
    """
    import sys

    global RUN_CONDITION_LABEL

    condition = str(
        condition
    ).strip()

    if condition not in RUN_CONDITION_CHOICES:
        raise ValueError(
            "Unknown run condition "
            f"{condition!r}; expected one of "
            f"{RUN_CONDITION_CHOICES}"
        )

    RUN_CONDITION_LABEL = condition

    for module_name in (
        "harvestloop.harvest",
        "harvestloop.runlog",
        "harvestloop.route",
        "harvestloop.metrics",
        "harvestloop.routing",
    ):
        module = sys.modules.get(
            module_name
        )

        if (
            module is not None
            and hasattr(
                module,
                "RUN_CONDITION_LABEL",
            )
        ):
            setattr(
                module,
                "RUN_CONDITION_LABEL",
                condition,
            )

    return RUN_CONDITION_LABEL


def apply_scene_identity(
    seed=None,
    occlusion=None,
    max_targets=None,
):
    """
    Record the scene identity this harvest run claims to match.

    Does not rebuild the greenhouse. The loaded USD must already
    have been generated with the same SEED / OCCLUSION_LEVEL.
    """
    import sys

    global SCENE_SEED
    global OCCLUSION_LEVEL
    global MAX_PAIRED_TARGETS

    if seed is not None:
        SCENE_SEED = int(seed)

    if occlusion is not None:
        occlusion = str(occlusion).strip().lower()

        if occlusion not in OCCLUSION_CHOICES:
            raise ValueError(
                "Unknown occlusion "
                f"{occlusion!r}; expected one of "
                f"{OCCLUSION_CHOICES}"
            )

        OCCLUSION_LEVEL = occlusion

    if max_targets is not None:
        if str(max_targets) in ("", "none", "None"):
            MAX_PAIRED_TARGETS = None
        else:
            MAX_PAIRED_TARGETS = int(max_targets)

    values = {
        "SCENE_SEED": SCENE_SEED,
        "OCCLUSION_LEVEL": OCCLUSION_LEVEL,
        "MAX_PAIRED_TARGETS": MAX_PAIRED_TARGETS,
    }

    for module_name in (
        "harvestloop.harvest",
        "harvestloop.runlog",
        "harvestloop.route",
        "harvestloop.metrics",
        "harvestloop.routing",
    ):
        module = sys.modules.get(module_name)

        if module is None:
            continue

        for key, value in values.items():
            if hasattr(module, key):
                setattr(module, key, value)

    return {
        "scene_seed": SCENE_SEED,
        "occlusion_level": OCCLUSION_LEVEL,
        "max_paired_targets": MAX_PAIRED_TARGETS,
    }


def _argv_flag_value(args, names):
    """Return value for --flag VALUE or --flag=VALUE, else None."""
    for index, arg in enumerate(
        args
    ):
        if arg in names:
            if index + 1 >= len(
                args
            ):
                raise ValueError(
                    f"{arg} requires a value"
                )

            return args[
                index + 1
            ]

        for name in names:
            prefix = f"{name}="

            if arg.startswith(
                prefix
            ):
                return arg[
                    len(
                        prefix
                    ):
                ]

    return None


def apply_run_experiment(
    experiment_id,
):
    """Set RUN_EXPERIMENT_ID for this process."""
    global RUN_EXPERIMENT_ID

    experiment_id = str(
        experiment_id
    ).strip()

    if not experiment_id:
        raise ValueError(
            "experiment id must be non-empty"
        )

    RUN_EXPERIMENT_ID = experiment_id

    return RUN_EXPERIMENT_ID


def _condition_dir_complete(
    experiment_dir,
    condition,
):
    return (
        experiment_dir
        / condition
        / "trusses.csv"
    ).exists()


def _newest_open_experiment(
    condition,
):
    """
    Newest experiment folder that still lacks this condition's
    finished trusses.csv — so fixed then active share a parent.
    """
    if not RUN_ARTIFACTS_ROOT.exists():
        return None

    candidates = []

    for path in RUN_ARTIFACTS_ROOT.iterdir():
        if not path.is_dir():
            continue

        # Skip legacy flat run dirs named <stamp>_<condition>.
        if path.name.endswith(
            f"_{condition}"
        ):
            continue

        looks_like_pair_parent = any(
            (
                path / choice
            ).is_dir()
            for choice in RUN_CONDITION_CHOICES
        ) or (
            path / "experiment.json"
        ).exists()

        # Brand-new empty timestamp folders also count once created.
        if not looks_like_pair_parent:
            # Accept bare YYYYMMDD_HHMMSS folders.
            parts = path.name.split(
                "_"
            )

            if not (
                len(
                    parts
                ) == 2
                and parts[0].isdigit()
                and len(
                    parts[0]
                ) == 8
                and parts[1].isdigit()
                and len(
                    parts[1]
                ) == 6
            ):
                continue

        if _condition_dir_complete(
            path,
            condition,
        ):
            continue

        candidates.append(
            path
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: path.name,
    ).name


def resolve_run_experiment(
    argv=None,
    explicit=None,
    condition=None,
):
    """
    Pick the parent experiment folder id for this run.

    Precedence:
      1. explicit=... (launcher / Script Editor EXPERIMENT)
      2. --experiment / -e on argv
      3. HARVESTLOOP_EXPERIMENT env var
      4. RUN_EXPERIMENT_ID if already set
      5. newest experiment folder still missing this condition
      6. new YYYYMMDD_HHMMSS stamp
    """
    import os
    import sys
    from datetime import datetime

    if explicit is not None:
        return apply_run_experiment(
            explicit
        )

    args = list(
        sys.argv[1:]
        if argv is None
        else argv
    )

    flagged = _argv_flag_value(
        args,
        (
            "--experiment",
            "-e",
        ),
    )

    if flagged is not None:
        return apply_run_experiment(
            flagged
        )

    env = os.environ.get(
        "HARVESTLOOP_EXPERIMENT"
    )

    if env:
        return apply_run_experiment(
            env
        )

    if RUN_EXPERIMENT_ID:
        return apply_run_experiment(
            RUN_EXPERIMENT_ID
        )

    if condition is not None:
        open_id = _newest_open_experiment(
            condition
        )

        if open_id is not None:
            return apply_run_experiment(
                open_id
            )

    return apply_run_experiment(
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )


def resolve_run_condition(
    argv=None,
    explicit=None,
):
    """
    Pick the viewpoint condition for this run.

    Precedence:
      1. explicit=... (launcher / Script Editor)
      2. --condition / -c on argv
      3. HARVESTLOOP_CONDITION env var
      4. RUN_CONDITION_LABEL default in this file
    """
    import os
    import sys

    if explicit is not None:
        return apply_run_condition(
            explicit
        )

    args = list(
        sys.argv[1:]
        if argv is None
        else argv
    )

    flagged = _argv_flag_value(
        args,
        (
            "--condition",
            "-c",
        ),
    )

    if flagged is not None:
        return apply_run_condition(
            flagged
        )

    env = os.environ.get(
        "HARVESTLOOP_CONDITION"
    )

    if env:
        return apply_run_condition(
            env
        )

    return RUN_CONDITION_LABEL
