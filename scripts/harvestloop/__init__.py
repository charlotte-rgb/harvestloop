"""
HarvestLoop aisle-1 harvesting route.

Layered so each module only imports from the ones above it:

    config        constants, USD paths, tuning
    mathutils     quaternion / angle math
    simctl        frame stepping, articulation init
    usd_pose      USD pose queries and frame conversions
    arm_control   UR5e joint control, RMPflow setup
    base_control  Jackal driving and waypoint following
    basket        basket geometry and PhysX preparation
    payload       truss carry ops, attach / release
    perception    wrist camera capture
    routing       stem discovery and stop planning
    manipulation  RMPflow stages (view, pregrasp, grasp, cut, retreat)
    harvest       per-truss cycle and per-stop orchestration
    route         main()

Run it from the Isaac Script Editor via
scripts/run_aisle1_viewpoint.py.
"""
