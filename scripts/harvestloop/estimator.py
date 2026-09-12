"""
Perception seam: target estimation for the harvest controller.

The manipulation stack must not read USD GraspPoint / CutPoint.
It only consumes:

    estimate_target(rgb, depth, camera_pose) -> {
        "grasp_point_world": (3,),
        "cut_point_world": (3,),
        "confidence": float,
        ...
    }

USD markers remain available as hidden ground truth for scoring.
The oracle estimator below still returns those values internally so
the wiring can be validated (localization error ≈ 0) before a real
RGB-D estimator is plugged in.
"""

import numpy as np

from .usd_pose import pose_position


def derive_grasp_from_cut(
    cut_point_world,
    peduncle_direction,
    tool_spacing_m,
):
    """
    GraspPoint from CutPoint using the fixed tool spacing.

    Scene generation authored:

        CutPoint = GraspPoint - peduncle_direction * spacing

    so the inverse is:

        GraspPoint = CutPoint + peduncle_direction * spacing
    """
    cut = np.asarray(
        cut_point_world,
        dtype=np.float64,
    ).reshape(3)

    direction = np.asarray(
        peduncle_direction,
        dtype=np.float64,
    ).reshape(3)

    norm = float(
        np.linalg.norm(
            direction
        )
    )

    if norm <= 1.0e-9:
        raise ValueError(
            "peduncle_direction has near-zero length."
        )

    direction = direction / norm

    return (
        cut
        + direction
        * float(
            tool_spacing_m
        )
    )


def measure_tool_tip_spacing(
    stage,
    grasp_tip_path,
    cutter_tip_path,
):
    """
    ||GraspTip - CutterTip|| from the live robot, in metres.
    """
    grasp_tip = pose_position(
        stage,
        grasp_tip_path,
    )

    cutter_tip = pose_position(
        stage,
        cutter_tip_path,
    )

    return float(
        np.linalg.norm(
            grasp_tip
            - cutter_tip
        )
    )


def localization_errors(
    estimate,
    gt_grasp_world,
    gt_cut_world,
):
    """
    Score an estimate against hidden USD GT.

    Returns metres; missing GT yields None for that field.
    """
    errors = {
        "grasp_error_m": None,
        "cut_error_m": None,
    }

    if (
        estimate is None
        or estimate.get(
            "grasp_point_world"
        ) is None
        or gt_grasp_world is None
    ):
        pass
    else:
        errors["grasp_error_m"] = float(
            np.linalg.norm(
                np.asarray(
                    estimate[
                        "grasp_point_world"
                    ],
                    dtype=np.float64,
                )
                - np.asarray(
                    gt_grasp_world,
                    dtype=np.float64,
                )
            )
        )

    if (
        estimate is None
        or estimate.get(
            "cut_point_world"
        ) is None
        or gt_cut_world is None
    ):
        pass
    else:
        errors["cut_error_m"] = float(
            np.linalg.norm(
                np.asarray(
                    estimate[
                        "cut_point_world"
                    ],
                    dtype=np.float64,
                )
                - np.asarray(
                    gt_cut_world,
                    dtype=np.float64,
                )
            )
        )

    return errors


class OracleUsdEstimator:
    """
    Temporary estimator that returns USD GT through the seam.

    `bind_truss` is called by the route orchestrator so this object
    knows which markers to read. The harvest controller still only
    sees `estimate_target(rgb, depth, camera_pose)` and must not
    open the USD paths itself.
    """

    def __init__(
        self,
        stage,
        tool_spacing_m=None,
    ):
        self.stage = stage
        self.tool_spacing_m = tool_spacing_m
        self._grasp_path = None
        self._cut_path = None

    @property
    def name(self):
        return "oracle_usd"

    def bind_truss(
        self,
        grasp_path,
        cut_path,
    ):
        self._grasp_path = grasp_path
        self._cut_path = cut_path

    def estimate_target(
        self,
        rgb,
        depth,
        camera_pose,
    ):
        """
        The controller-facing interface.

        rgb / depth / camera_pose are accepted and ignored while this
        oracle is verifying the wiring. A real estimator will use
        them and never receive USD paths.
        """
        _ = (
            rgb,
            depth,
            camera_pose,
        )

        if (
            self._grasp_path is None
            or self._cut_path is None
        ):
            raise RuntimeError(
                "OracleUsdEstimator.bind_truss(...) must be "
                "called before estimate_target(...)."
            )

        grasp = pose_position(
            self.stage,
            self._grasp_path,
        )

        cut = pose_position(
            self.stage,
            self._cut_path,
        )

        # Exercise the same CutPoint -> GraspPoint derivation the
        # vision pipeline will use, so a spacing bug shows up here.
        peduncle = grasp - cut

        if self.tool_spacing_m is None:
            spacing = float(
                np.linalg.norm(
                    peduncle
                )
            )
        else:
            spacing = float(
                self.tool_spacing_m
            )

        derived_grasp = derive_grasp_from_cut(
            cut,
            peduncle,
            spacing,
        )

        return {
            "grasp_point_world": np.asarray(
                derived_grasp,
                dtype=np.float64,
            ),
            "cut_point_world": np.asarray(
                cut,
                dtype=np.float64,
            ),
            "confidence": 1.0,
            "source": self.name,
            "views_used": 1,
            "tool_spacing_m": spacing,
        }


def make_estimator(
    stage,
    mode="oracle_usd",
    tool_spacing_m=None,
):
    """
    Factory for the single perception seam used by the route.
    """
    if mode == "oracle_usd":
        return OracleUsdEstimator(
            stage,
            tool_spacing_m=tool_spacing_m,
        )

    if mode == "rgbd_peduncle":
        from .rgbd_estimate import RgbdPeduncleEstimator

        return RgbdPeduncleEstimator(
            stage,
            tool_spacing_m=tool_spacing_m,
        )

    if mode == "keypoints":
        from .keypoint_estimate import KeypointTargetEstimator

        return KeypointTargetEstimator(
            stage,
            tool_spacing_m=tool_spacing_m,
        )

    raise ValueError(
        f"Unknown estimator mode: {mode!r}"
    )
