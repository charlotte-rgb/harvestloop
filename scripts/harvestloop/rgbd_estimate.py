"""
RGB-D peduncle target estimation for the fixed-view baseline.

Pipeline:
  1. Segment tomato (red) and peduncle (light green) in RGB
  2. Keep peduncle pixels near the tomato cluster with valid depth
  3. Back-project to 3D and fit a line
  4. CutPoint = proximal end of the visible peduncle segment
  5. GraspPoint = CutPoint + peduncle_direction * tool_spacing

No USD GraspPoint / CutPoint is read here.
"""

import numpy as np

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover
    ndimage = None


def _quaternion_to_rotation_matrix(q):
    q = np.asarray(
        q,
        dtype=np.float64,
    ).reshape(4)

    n = float(
        np.linalg.norm(
            q
        )
    )

    if n < 1.0e-12:
        w, x, y, z = (
            1.0,
            0.0,
            0.0,
            0.0,
        )
    else:
        w, x, y, z = q / n

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _derive_grasp_from_cut(
    cut_point_world,
    peduncle_direction,
    tool_spacing_m,
):
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


# Authored peduncle / tomato colours in scenegen/params.py
# (linear-ish preview values). Rendered pixels land nearby.
_PEDUNCLE_RGB = np.array(
    [0.28, 0.58, 0.14],
    dtype=np.float64,
)

_TOMATO_RGB = np.array(
    [0.78, 0.055, 0.035],
    dtype=np.float64,
)

# Depth band around the canonical wrist standoff. Rejects the
# near-field robot body and far background / sky.
_DEPTH_MIN_M = 0.12
_DEPTH_MAX_M = 0.55

_MIN_PEDUNCLE_PIXELS = 30
_MIN_TOMATO_PIXELS = 15
_MAX_FIT_POINTS = 2500

# Soft confidence thresholds.
_RESIDUAL_GOOD_M = 0.008
_RESIDUAL_BAD_M = 0.035
_BORDER_MARGIN_PX = 24


def _as_rgb01(rgb):
    rgb = np.asarray(
        rgb
    )

    if rgb.dtype == np.uint8:
        return rgb.astype(
            np.float64
        ) / 255.0

    rgb = rgb.astype(
        np.float64
    )

    if float(
        np.nanmax(
            rgb
        )
    ) > 1.5:
        return rgb / 255.0

    return rgb


def _tomato_mask(rgb01):
    r = rgb01[
        :,
        :,
        0,
    ]
    g = rgb01[
        :,
        :,
        1,
    ]
    b = rgb01[
        :,
        :,
        2,
    ]

    return (
        (r > 0.35)
        & (r > g + 0.12)
        & (r > b + 0.12)
        & (g < 0.45)
        & (b < 0.45)
    )


def _peduncle_mask(rgb01):
    """
    Light / medium green, not dark leaf and not tomato.
    """
    r = rgb01[
        :,
        :,
        0,
    ]
    g = rgb01[
        :,
        :,
        1,
    ]
    b = rgb01[
        :,
        :,
        2,
    ]

    dist = np.sqrt(
        (r - _PEDUNCLE_RGB[0]) ** 2
        + (g - _PEDUNCLE_RGB[1]) ** 2
        + (b - _PEDUNCLE_RGB[2]) ** 2
    )

    # Exclude dark main-stem / leaf greens.
    green_dom = (
        (g > r + 0.07)
        & (g > b + 0.05)
        & (g > 0.30)
        & (g < 0.72)
        & (r > 0.10)
        & (r < 0.42)
        & (b < 0.42)
    )

    near_peduncle = dist < 0.32

    return green_dom & near_peduncle


def _valid_depth(depth):
    depth = np.asarray(
        depth,
        dtype=np.float64,
    )

    if depth.ndim == 3:
        depth = depth[
            :,
            :,
            0,
        ]

    return (
        np.isfinite(
            depth
        )
        & (depth >= _DEPTH_MIN_M)
        & (depth <= _DEPTH_MAX_M)
    ), depth


def _largest_component(mask):
    if not np.any(
        mask
    ):
        return mask

    if ndimage is None:
        return mask

    labeled, count = ndimage.label(
        mask.astype(
            np.uint8
        )
    )

    if count <= 0:
        return mask

    sizes = ndimage.sum(
        mask,
        labeled,
        index=np.arange(
            1,
            count + 1,
        ),
    )

    best = int(
        np.argmax(
            sizes
        )
    ) + 1

    return labeled == best


def _dilate(mask, iterations=8):
    if ndimage is None:
        return mask

    return ndimage.binary_dilation(
        mask,
        iterations=int(
            iterations
        ),
    )


def backproject_pixels(
    us,
    vs,
    depths,
    intrinsics,
):
    """
    Pixel + distance_to_image_plane -> points in USD camera frame.

    Matches perception.project_world_point_to_wrist_image:
    camera looks down -Z, +X right, +Y up.
    """
    us = np.asarray(
        us,
        dtype=np.float64,
    )
    vs = np.asarray(
        vs,
        dtype=np.float64,
    )
    depths = np.asarray(
        depths,
        dtype=np.float64,
    )

    width = float(
        intrinsics["width"]
    )
    height = float(
        intrinsics["height"]
    )
    focal = float(
        intrinsics["focal_length"]
    )
    half_aperture = 0.5 * float(
        intrinsics[
            "horizontal_aperture"
        ]
    )
    half_vertical = (
        half_aperture
        * height
        / width
    )

    x_ndc = (
        2.0 * us / width
    ) - 1.0

    y_ndc = 1.0 - (
        2.0 * vs / height
    )

    x = (
        x_ndc
        * half_aperture
        * depths
        / focal
    )
    y = (
        y_ndc
        * half_vertical
        * depths
        / focal
    )
    z = -depths

    return np.column_stack(
        (
            x,
            y,
            z,
        )
    )


def _project_camera_point_to_pixel(
    point_in_camera,
    intrinsics,
):
    depth = -float(
        point_in_camera[2]
    )

    if depth <= 1.0e-6:
        return None

    width = float(
        intrinsics["width"]
    )
    height = float(
        intrinsics["height"]
    )
    half_aperture = 0.5 * float(
        intrinsics[
            "horizontal_aperture"
        ]
    )

    scale = float(
        intrinsics["focal_length"]
    ) / depth

    x_ndc = (
        scale
        * float(
            point_in_camera[0]
        )
        / half_aperture
    )

    y_ndc = (
        scale
        * float(
            point_in_camera[1]
        )
        / (
            half_aperture
            * height
            / width
        )
    )

    return np.array(
        [
            (0.5 + 0.5 * x_ndc) * width,
            (0.5 - 0.5 * y_ndc) * height,
        ],
        dtype=np.float64,
    )


def camera_to_world(
    points_cam,
    camera_pose,
):
    R = _quaternion_to_rotation_matrix(
        camera_pose[
            "orientation"
        ]
    )

    t = np.asarray(
        camera_pose[
            "position"
        ],
        dtype=np.float64,
    ).reshape(3)

    points_cam = np.asarray(
        points_cam,
        dtype=np.float64,
    ).reshape(
        -1,
        3,
    )

    return (
        points_cam @ R.T
    ) + t


def fit_line_3d(points):
    """
    PCA line through 3D points.

    Returns (centroid, unit direction, rms residual metres).
    """
    points = np.asarray(
        points,
        dtype=np.float64,
    ).reshape(
        -1,
        3,
    )

    centroid = points.mean(
        axis=0
    )

    centered = points - centroid

    # Covariance eigen-decomposition; largest eigenvector is the axis.
    cov = (
        centered.T @ centered
    ) / max(
        len(
            points
        ),
        1,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        cov
    )

    direction = eigenvectors[
        :,
        int(
            np.argmax(
                eigenvalues
            )
        ),
    ]

    norm = float(
        np.linalg.norm(
            direction
        )
    )

    if norm <= 1.0e-12:
        raise ValueError(
            "Degenerate peduncle line."
        )

    direction = direction / norm

    residuals = np.linalg.norm(
        centered
        - np.outer(
            centered @ direction,
            direction,
        ),
        axis=1,
    )

    rms = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    return centroid, direction, rms


def _confidence(
    *,
    n_points,
    peduncle_pixel_count,
    valid_depth_fraction,
    residual_m,
    visible_length_m,
    tool_spacing_m,
    cut_uv,
    image_shape,
):
    point_score = min(
        1.0,
        n_points / 120.0,
    )

    depth_score = float(
        np.clip(
            valid_depth_fraction,
            0.0,
            1.0,
        )
    )

    if residual_m <= _RESIDUAL_GOOD_M:
        residual_score = 1.0
    elif residual_m >= _RESIDUAL_BAD_M:
        residual_score = 0.0
    else:
        residual_score = 1.0 - (
            (residual_m - _RESIDUAL_GOOD_M)
            / (
                _RESIDUAL_BAD_M
                - _RESIDUAL_GOOD_M
            )
        )

    # Expect to see at least the tool spacing on the peduncle.
    length_score = float(
        np.clip(
            visible_length_m
            / max(
                tool_spacing_m,
                1.0e-3,
            ),
            0.0,
            1.0,
        )
    )

    border_score = 1.0

    if (
        cut_uv is not None
        and image_shape is not None
    ):
        h, w = image_shape[
            :2
        ]
        u, v = cut_uv
        margin = _BORDER_MARGIN_PX

        dist_border = min(
            u,
            v,
            w - 1 - u,
            h - 1 - v,
        )

        border_score = float(
            np.clip(
                dist_border / margin,
                0.0,
                1.0,
            )
        )

    # Mild weight on raw peduncle coverage so empty frames score low.
    coverage_score = min(
        1.0,
        peduncle_pixel_count / 200.0,
    )

    scores = np.array(
        [
            point_score,
            depth_score,
            residual_score,
            length_score,
            border_score,
            coverage_score,
        ],
        dtype=np.float64,
    )

    return float(
        np.clip(
            scores.mean(),
            0.0,
            1.0,
        )
    )


def estimate_peduncle_targets(
    rgb,
    depth,
    camera_pose,
    tool_spacing_m,
):
    """
    Core RGB-D estimate. Returns the estimate_target payload.
    """
    empty = {
        "grasp_point_world": None,
        "cut_point_world": None,
        "confidence": 0.0,
        "source": "rgbd_peduncle",
        "views_used": 1,
        "tool_spacing_m": float(
            tool_spacing_m
        ),
        "diagnostics": {},
    }

    if (
        rgb is None
        or depth is None
        or camera_pose is None
    ):
        empty["diagnostics"] = {
            "reason": "missing_rgb_depth_or_pose"
        }
        return empty

    rgb01 = _as_rgb01(
        rgb
    )
    valid, depth = _valid_depth(
        depth
    )

    tomato = _tomato_mask(
        rgb01
    )
    peduncle = _peduncle_mask(
        rgb01
    )

    tomato_count = int(
        tomato.sum()
    )
    peduncle_count = int(
        peduncle.sum()
    )

    tomato_world = None

    # Prefer peduncle near tomatoes, gated to the tomato depth band
    # so the main stem / far plants do not dominate the fit.
    if tomato_count >= _MIN_TOMATO_PIXELS:
        near_tomato = _dilate(
            tomato,
            iterations=10,
        )
        tomato_depth = depth[
            tomato & valid
        ]

        if tomato_depth.size > 0:
            d_lo = float(
                np.percentile(
                    tomato_depth,
                    10,
                )
            ) - 0.06
            d_hi = float(
                np.percentile(
                    tomato_depth,
                    90,
                )
            ) + 0.06

            depth_band = (
                (depth >= d_lo)
                & (depth <= d_hi)
            )
        else:
            depth_band = valid

        candidate = (
            peduncle
            & near_tomato
            & valid
            & depth_band
        )
    else:
        candidate = peduncle & valid

    candidate = _largest_component(
        candidate
    )

    if int(
        candidate.sum()
    ) < _MIN_PEDUNCLE_PIXELS:
        candidate = _largest_component(
            peduncle & valid
        )

    ys, xs = np.where(
        candidate
    )

    diagnostics = {
        "tomato_pixels": tomato_count,
        "peduncle_pixels": peduncle_count,
        "candidate_pixels": int(
            len(
                xs
            )
        ),
        "valid_depth_fraction": float(
            (
                peduncle & valid
            ).sum()
            / max(
                peduncle_count,
                1,
            )
        ),
    }

    if len(
        xs
    ) < _MIN_PEDUNCLE_PIXELS:
        empty["diagnostics"] = {
            **diagnostics,
            "reason": "too_few_peduncle_pixels",
        }
        return empty

    depths = depth[
        ys,
        xs,
    ]

    # Reject depth speckles (robot body / holes) that survived the
    # global band: keep a tight window around the candidate median.
    depth_med = float(
        np.median(
            depths
        )
    )

    depth_keep = np.abs(
        depths - depth_med
    ) <= 0.07

    if int(
        depth_keep.sum()
    ) >= _MIN_PEDUNCLE_PIXELS:
        xs = xs[
            depth_keep
        ]
        ys = ys[
            depth_keep
        ]
        depths = depths[
            depth_keep
        ]

    if len(
        xs
    ) > _MAX_FIT_POINTS:
        idx = np.linspace(
            0,
            len(xs) - 1,
            _MAX_FIT_POINTS,
        ).astype(
            np.int64
        )
        xs = xs[
            idx
        ]
        ys = ys[
            idx
        ]
        depths = depths[
            idx
        ]

    points_cam = backproject_pixels(
        xs,
        ys,
        depths,
        camera_pose[
            "intrinsics"
        ],
    )

    points_world = camera_to_world(
        points_cam,
        camera_pose,
    )

    try:
        centroid, direction, residual_m = fit_line_3d(
            points_world
        )
    except ValueError as exc:
        empty["diagnostics"] = {
            **diagnostics,
            "reason": str(
                exc
            ),
        }
        return empty

    # If the first fit mixes stem + peduncle, keep only inliers.
    if residual_m > 0.015:
        centered = points_world - centroid
        point_resid = np.linalg.norm(
            centered
            - np.outer(
                centered @ direction,
                direction,
            ),
            axis=1,
        )
        inliers = point_resid <= max(
            0.012,
            1.5 * np.median(
                point_resid
            ),
        )

        if int(
            inliers.sum()
        ) >= _MIN_PEDUNCLE_PIXELS:
            points_world = points_world[
                inliers
            ]
            try:
                centroid, direction, residual_m = fit_line_3d(
                    points_world
                )
            except ValueError:
                pass

    # Orient direction toward the tomato cluster (distal).
    if tomato_count >= _MIN_TOMATO_PIXELS:
        ty, tx = np.where(
            tomato & valid
        )

        if len(
            tx
        ) > 0:
            step = max(
                1,
                len(tx) // 800,
            )
            tomato_cam = backproject_pixels(
                tx[
                    ::step
                ],
                ty[
                    ::step
                ],
                depth[
                    ty[
                        ::step
                    ],
                    tx[
                        ::step
                    ],
                ],
                camera_pose[
                    "intrinsics"
                ],
            )

            tomato_world = camera_to_world(
                tomato_cam,
                camera_pose,
            ).mean(
                axis=0
            )

            if float(
                np.dot(
                    tomato_world - centroid,
                    direction,
                )
            ) < 0.0:
                direction = -direction

    projections = (
        points_world - centroid
    ) @ direction

    t_min = float(
        projections.min()
    )
    t_max = float(
        projections.max()
    )

    visible_length_m = max(
        0.0,
        t_max - t_min,
    )

    # Grasp sits on the distal (tomato-side) peduncle; CutPoint is
    # one tool-spacing step back toward the stem.
    if tomato_world is not None:
        tomato_t = float(
            np.dot(
                tomato_world - centroid,
                direction,
            )
        )
        # Peduncle samples near the tomatoes.
        distal = projections >= (
            0.55 * t_min
            + 0.45 * max(
                t_max,
                tomato_t,
            )
        )

        if int(
            distal.sum()
        ) < 5:
            distal = projections >= (
                t_min
                + 0.60 * visible_length_m
            )
    else:
        distal = projections >= (
            t_min
            + 0.60 * visible_length_m
        )

    if int(
        distal.sum()
    ) < 3:
        t_grasp = t_max
    else:
        t_grasp = float(
            np.median(
                projections[
                    distal
                ]
            )
        )

    grasp_world = (
        centroid
        + direction * t_grasp
    )

    cut_world = (
        grasp_world
        - direction
        * float(
            tool_spacing_m
        )
    )

    # Keep the public derive helper path consistent.
    grasp_world = _derive_grasp_from_cut(
        cut_world,
        direction,
        tool_spacing_m,
    )

    # Project cut for border confidence (camera frame).
    R = _quaternion_to_rotation_matrix(
        camera_pose[
            "orientation"
        ]
    )

    cut_cam = R.T @ (
        cut_world
        - np.asarray(
            camera_pose[
                "position"
            ],
            dtype=np.float64,
        ).reshape(3)
    )

    cut_uv = _project_camera_point_to_pixel(
        cut_cam,
        camera_pose[
            "intrinsics"
        ],
    )

    confidence = _confidence(
        n_points=len(
            xs
        ),
        peduncle_pixel_count=peduncle_count,
        valid_depth_fraction=diagnostics[
            "valid_depth_fraction"
        ],
        residual_m=residual_m,
        visible_length_m=visible_length_m,
        tool_spacing_m=float(
            tool_spacing_m
        ),
        cut_uv=cut_uv,
        image_shape=rgb01.shape,
    )

    # Extra penalties for geometries that cannot be a short peduncle
    # next to a tomato cluster — keeps confidence honest for the
    # later active-perception gate.
    if residual_m > 0.012:
        confidence *= 0.55

    if tomato_world is not None:
        cut_to_tomato = float(
            np.linalg.norm(
                cut_world - tomato_world
            )
        )

        diagnostics[
            "cut_to_tomato_m"
        ] = cut_to_tomato

        if cut_to_tomato > 0.22:
            confidence *= 0.35

        if cut_to_tomato < 0.01:
            confidence *= 0.5

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )

    diagnostics.update(
        {
            "line_residual_m": residual_m,
            "visible_length_m": visible_length_m,
            "cut_uv": (
                None
                if cut_uv is None
                else [
                    float(
                        cut_uv[0]
                    ),
                    float(
                        cut_uv[1]
                    ),
                ]
            ),
        }
    )

    return {
        "grasp_point_world": np.asarray(
            grasp_world,
            dtype=np.float64,
        ),
        "cut_point_world": np.asarray(
            cut_world,
            dtype=np.float64,
        ),
        "confidence": confidence,
        "source": "rgbd_peduncle",
        "views_used": 1,
        "tool_spacing_m": float(
            tool_spacing_m
        ),
        "diagnostics": diagnostics,
    }


class RgbdPeduncleEstimator:
    """
    Fixed-view RGB-D estimator behind estimate_target(...).
    """

    def __init__(
        self,
        stage=None,
        tool_spacing_m=None,
    ):
        _ = stage
        if (
            tool_spacing_m is None
            or float(
                tool_spacing_m
            ) <= 0.0
        ):
            raise ValueError(
                "RgbdPeduncleEstimator requires a positive "
                "tool_spacing_m."
            )

        self.tool_spacing_m = float(
            tool_spacing_m
        )

    @property
    def name(self):
        return "rgbd_peduncle"

    def estimate_target(
        self,
        rgb,
        depth,
        camera_pose,
    ):
        return estimate_peduncle_targets(
            rgb=rgb,
            depth=depth,
            camera_pose=camera_pose,
            tool_spacing_m=self.tool_spacing_m,
        )
