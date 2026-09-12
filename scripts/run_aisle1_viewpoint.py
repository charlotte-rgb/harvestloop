# Short launcher for Isaac Script Editor.
# Paste/run THIS file to avoid path truncation in the editor stub.
# Full aisle-1 route: viewpoint + grasp/cut/carry/basket release.
#
# Layout:
#   results/runs/<experiment_id>/
#       fixed_view/
#       active_perception/
#
# Preferred Script Editor usage:
#   CONDITION = "fixed_view"          # or "active_perception"
#   # EXPERIMENT = "20260909_192714"  # optional; auto-pairs if omitted
#   exec(open("/home/charlotte/harvestloop_sim/scripts/run_aisle1_viewpoint.py", encoding="utf-8").read(), globals())
#
# First condition creates <experiment_id>. Second condition reuses the
# newest experiment folder that still lacks that condition — so fixed
# then active share one parent without setting EXPERIMENT.
#
# Never assign CONDITION/EXPERIMENT = None here when already set —
# that clobbers paste-before-exec overrides.
if "CONDITION" not in globals():
    CONDITION = None

if "EXPERIMENT" not in globals():
    EXPERIMENT = None

exec(
    open(
        "/home/charlotte/harvestloop_sim/scripts/"
        "harvest_aisle1_route_viewpoint_test.py",
        encoding="utf-8",
    ).read(),
    globals(),
)
