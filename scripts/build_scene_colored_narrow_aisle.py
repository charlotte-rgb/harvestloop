# =========================================================
# build_scene_colored_narrow_aisle.py
#
# Narrow first-aisle greenhouse scene: the cross gap is small
# enough for the Jackal/UR5e to reach both left and right beds.
#
# Baseline CROSS_AISLE_GAP = 0.80 m, BED_WIDTH = 0.50 * CROP_SCALE
# This scene:  CROSS_AISLE_GAP = 0.65 m, BED_WIDTH = 0.35 * CROP_SCALE
#
# Narrower beds + moderately wider aisle free space so the Jackal can
# stay centered while the scaled UR5e still reaches both sides.
#
# Scene contents live in the scenegen package; tunables are in
# scenegen/params.py.
# =========================================================

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False
})

# Kit must be up before anything imports omni/pxr, so the scene
# package is imported only after SimulationApp exists.
import os

from scenegen.builder import build_scene
from scenegen.params import apply_occlusion_level, apply_scene_seed

seed = os.environ.get("HARVESTLOOP_SEED")
if seed is not None:
    apply_scene_seed(seed)

occlusion = os.environ.get("HARVESTLOOP_OCCLUSION")
if occlusion is not None:
    apply_occlusion_level(occlusion)

world = build_scene()


while simulation_app.is_running():
    world.step(
        render=True
    )


simulation_app.close()
