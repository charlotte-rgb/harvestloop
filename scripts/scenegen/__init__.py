"""
Narrow-aisle greenhouse scene generation.

Layered so each module only imports from the ones above it:

    params      bed/aisle geometry, colours, crop scale, USD paths
    materials   preview materials and binding helpers
    geometry    boxes, spheres, cylinders and vector helpers
    tooling     GraspTip <-> CutterTip spacing measured from the robot USD
    plants      leaves, branches, trusses, main stems
    layout      crop beds and greenhouse rows
    builder     build_scene()

Built by scripts/build_scene_colored_narrow_aisle.py, which must
create the SimulationApp before importing this package.
"""
