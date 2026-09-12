# Control backbone checkpoint

**Tag:** `backbone-control-v1`

Frozen as the experimental control backbone. From this point, do not
change navigation, grasp/cut tolerances, retreat logic, basket logic,
or route structure unless there is a clear bug.

## What is frozen

- Aisle-1 route structure and stop planning
- Jackal drive / recenter behaviour
- Aimed wrist viewpoint placement (canonical fixed view)
- RMPflow grasp / cut / retreat / upright / basket sequence
- GraspTip ≤ 3 cm and CutterTip ≤ 4 cm hard tolerances
- Post-CUT retreat retries and basket containment

## What is allowed to change

- The perception estimator behind `estimate_target(...)`
- RGB-D capture and labelling used only for estimation / scoring
- Logging fields that score estimates against hidden USD GT
- Later: an active-perception policy that chooses additional views
  while keeping the same estimator and the same manipulation
  controller

## Experiment contract

The manipulation controller consumes only:

```text
estimate_target(rgb, depth, camera_pose) ->
    grasp_point_world, cut_point_world, confidence
```

USD GraspPoint / CutPoint remain available as hidden ground truth for
scoring. They must not be read by the harvest controller after the
estimator returns.
