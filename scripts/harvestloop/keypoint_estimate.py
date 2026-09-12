"""
Keypoint-based target estimation (fixed-view perception baseline).

Two task keypoints per truss, both on the peduncle:

    K1  CutPoint   (stem-proximal task point)
    K2  GraspPoint (distal of K1 by the tool tip spacing)

Pipeline:

    1. find the peduncle attached to this truss's fruit
    2. place K1 / K2 on that axis at the authored cut / grasp offsets
       from the stem end
    3. lift each with the depth around its pixel
    4. CutPoint = K1, GraspPoint = K2
    5. peduncle_direction = normalize(K2 - K1)
    6. report a per-keypoint confidence so a later active-perception
       policy can ask for another view instead of harvesting blind

No USD is read here. Ground truth stays in the caller, for scoring.

Placement matches scenegen/plants.py:

    K1 = peduncle_start + dir * cut_offset
    K2 = K1             + dir * tool_spacing
"""

import numpy as np

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover
    ndimage = None

from .config import *  # noqa: F401,F403
from .rgbd_estimate import (
    _project_camera_point_to_pixel,
    _quaternion_to_rotation_matrix,
    backproject_pixels,
    camera_to_world,
    fit_line_3d,
)


# Measured palette of the rendered wrist view, NOT the authored
# material colours: lighting lifts everything, so the peduncle that
# is authored as medium green renders pale yellow-green.
#
#   peduncle   ~ (0.78, 0.90, 0.72)   pale, all channels high
#   main stem  ~ (0.40, 0.75, 0.42)   saturated green, r well below g
#   leaf       ~ (0.12, 0.33, 0.12)   dark green
#   tomato     ~ (0.94, 0.42, 0.30)   red
#
# The masks below are written as channel RATIOS rather than absolute
# thresholds, so a peduncle in shadow (every channel scaled down)
# still matches while the main stem never does.


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


def _channels(rgb01):
    return (
        rgb01[:, :, 0],
        rgb01[:, :, 1],
        rgb01[:, :, 2],
    )


def tomato_mask(rgb01):
    """Red fruit, at any brightness."""
    r, g, b = _channels(
        rgb01
    )

    return (
        (r > 0.30)
        & (g < 0.70 * r)
        & (b < 0.65 * r)
        & (r > g + 0.15)
    )


def peduncle_mask(rgb01):
    """
    Pale yellow-green peduncle, excluding the saturated main stem.

    r/g is the discriminator: about 0.85 on the peduncle and about
    0.53 on the stem, and it survives shading.
    """
    r, g, b = _channels(
        rgb01
    )

    return (
        (g > 0.28)
        & (g > r + 0.03)
        & (g > b + 0.04)
        & (r > 0.63 * g)
        & (b > 0.45 * g)
    )


def valid_depth_mask(depth):
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

    valid = (
        np.isfinite(
            depth
        )
        & (depth >= KEYPOINT_DEPTH_MIN)
        & (depth <= KEYPOINT_DEPTH_MAX)
    )

    return valid, depth


def _largest_component(mask):
    if (
        ndimage is None
        or not np.any(
            mask
        )
    ):
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

    return labeled == (
        int(
            np.argmax(
                sizes
            )
        )
        + 1
    )


def _dilate(
    mask,
    iterations,
):
    if ndimage is None:
        return mask

    return ndimage.binary_dilation(
        mask,
        iterations=int(
            iterations
        ),
    )


def _close_gaps(mask):
    """
    Bridge the one-pixel gaps antialiasing leaves along a thin
    peduncle, so it labels as one component instead of hundreds.
    """
    if ndimage is None:
        return mask

    return ndimage.binary_closing(
        mask,
        structure=np.ones(
            (3, 3),
            dtype=bool,
        ),
        iterations=1,
    )


def _component_touching(
    mask,
    anchor,
):
    """
    The connected blob of `mask` most attached to `anchor`.

    Selecting a whole component and then discarding the ones that do
    not touch the fruit keeps the FULL peduncle. Intersecting with the
    fruit neighbourhood instead would clip the peduncle to a stub near
    the tomatoes, which puts K1 nowhere near the stem.
    """
    if (
        ndimage is None
        or not np.any(
            mask
        )
    ):
        return mask

    labeled, count = ndimage.label(
        mask.astype(
            np.uint8
        )
    )

    if count <= 0:
        return mask

    best_label = 0
    best_score = 0

    for label in range(
        1,
        count + 1,
    ):
        component = labeled == label

        if int(
            component.sum()
        ) < KEYPOINT_MIN_PEDUNCLE_PIXELS:
            continue

        overlap = int(
            np.count_nonzero(
                component & anchor
            )
        )

        if overlap <= 0:
            continue

        # Prefer attachment to the fruit, then size, so a large
        # background blob brushing the anchor cannot win outright.
        score = overlap * 1000 + int(
            component.sum()
        )

        if score > best_score:
            best_score = score
            best_label = label

    if best_label == 0:
        return np.zeros_like(
            mask
        )

    return labeled == best_label


def _patch_depth(
    depth,
    valid,
    u,
    v,
    radius=None,
):
    """
    Median depth in a small window around a keypoint pixel.

    Returns (depth_m, valid_fraction). Depth is None when the window
    holds no usable sample, which is one of the conditions a later
    active-perception policy reacts to.
    """
    if radius is None:
        radius = KEYPOINT_DEPTH_PATCH_RADIUS

    height, width = depth.shape[
        :2
    ]

    u0 = int(
        np.clip(
            round(u) - radius,
            0,
            width - 1,
        )
    )
    u1 = int(
        np.clip(
            round(u) + radius + 1,
            1,
            width,
        )
    )
    v0 = int(
        np.clip(
            round(v) - radius,
            0,
            height - 1,
        )
    )
    v1 = int(
        np.clip(
            round(v) + radius + 1,
            1,
            height,
        )
    )

    window = depth[
        v0:v1,
        u0:u1,
    ]
    window_valid = valid[
        v0:v1,
        u0:u1,
    ]

    if window.size == 0:
        return None, 0.0

    fraction = float(
        window_valid.mean()
    )

    samples = window[
        window_valid
    ]

    if samples.size == 0:
        return None, fraction

    return (
        float(
            np.median(
                samples
            )
        ),
        fraction,
    )


def project_world_to_pixel(
    point_world,
    camera_pose,
):
    """
    Pixel for a world point, using the wrist camera's aperture fit.

    For overlays and scoring only; the estimator does not project.
    """
    rotation = _quaternion_to_rotation_matrix(
        camera_pose[
            "orientation"
        ]
    )

    point_camera = rotation.T @ (
        np.asarray(
            point_world,
            dtype=np.float64,
        ).reshape(3)
        - np.asarray(
            camera_pose[
                "position"
            ],
            dtype=np.float64,
        ).reshape(3)
    )

    return _project_camera_point_to_pixel(
        point_camera,
        camera_pose[
            "intrinsics"
        ],
    )


def _border_score(
    u,
    v,
    image_shape,
):
    height, width = image_shape[
        :2
    ]

    distance = min(
        u,
        v,
        width - 1 - u,
        height - 1 - v,
    )

    return float(
        np.clip(
            distance
            / float(
                KEYPOINT_BORDER_MARGIN_PX
            ),
            0.0,
            1.0,
        )
    )


def _keypoint_confidence(
    *,
    depth_valid_fraction,
    support_pixels,
    border_score,
    line_offset_m,
):
    """
    How much this one keypoint can be trusted, in [0, 1].

    Built from the measurable criteria an active-perception policy
    needs: is there depth here, is there enough evidence here, is it
    inside the frame, and does it agree with the peduncle axis.
    """
    support_score = float(
        np.clip(
            support_pixels
            / float(
                KEYPOINT_SUPPORT_TARGET_PIXELS
            ),
            0.0,
            1.0,
        )
    )

    if line_offset_m is None:
        agreement_score = 0.0
    else:
        agreement_score = float(
            np.clip(
                1.0
                - (
                    line_offset_m
                    / KEYPOINT_LINE_OFFSET_BAD_M
                ),
                0.0,
                1.0,
            )
        )

    return float(
        np.clip(
            np.mean(
                [
                    float(
                        np.clip(
                            depth_valid_fraction,
                            0.0,
                            1.0,
                        )
                    ),
                    support_score,
                    border_score,
                    agreement_score,
                ]
            ),
            0.0,
            1.0,
        )
    )


def _failure(
    reason,
    diagnostics=None,
    tool_spacing_m=0.0,
):
    """
    An estimate that declines to guess.

    `needs_another_view` is True on every failure: with one view the
    truss is skipped, and the active condition will use the same flag
    to acquire another view instead.
    """
    return {
        "grasp_point_world": None,
        "cut_point_world": None,
        "confidence": 0.0,
        "source": "keypoints",
        "views_used": 1,
        "tool_spacing_m": float(
            tool_spacing_m
        ),
        "keypoints": {
            "k1": None,
            "k2": None,
        },
        "needs_another_view": True,
        "reason": reason,
        "diagnostics": {
            **(
                diagnostics
                or {}
            ),
            "reason": reason,
        },
    }


def _fruit_cluster_candidates(
    tomato_valid,
    depth,
    intrinsics,
    camera_pose,
    max_candidates=None,
):
    """
    Ranked tomato clusters near the image centre.

    Returns a list of (mask, centroid_world, score), best first.
    """
    if max_candidates is None:
        max_candidates = KEYPOINT_FRUIT_CANDIDATES

    ys, xs = np.where(
        tomato_valid
    )

    if len(
        xs
    ) == 0:
        return []

    points_world = camera_to_world(
        backproject_pixels(
            xs,
            ys,
            depth[
                ys,
                xs,
            ],
            intrinsics,
        ),
        camera_pose,
    )

    height, width = tomato_valid.shape[
        :2
    ]

    center = np.array(
        [
            0.5 * width,
            0.5 * height,
        ],
        dtype=np.float64,
    )

    candidates = []

    if ndimage is None:
        if len(
            xs
        ) < KEYPOINT_MIN_FRUIT_PIXELS:
            return []

        mask = np.zeros_like(
            tomato_valid
        )

        mask[
            ys,
            xs,
        ] = True

        return [
            (
                mask,
                points_world.mean(
                    axis=0
                ),
                float(
                    len(
                        xs
                    )
                ),
            )
        ]

    labeled, count = ndimage.label(
        tomato_valid.astype(
            np.uint8
        )
    )

    for label in range(
        1,
        count + 1,
    ):
        member = labeled[
            ys,
            xs,
        ] == label

        size = int(
            member.sum()
        )

        if size < KEYPOINT_MIN_FRUIT_PIXELS:
            continue

        distance = float(
            np.linalg.norm(
                np.array(
                    [
                        xs[member].mean(),
                        ys[member].mean(),
                    ]
                )
                - center
            )
        )

        score = size / (
            1.0
            + distance / 80.0
        )

        seed_centroid = points_world[
            member
        ].mean(
            axis=0
        )

        nearby = np.linalg.norm(
            points_world - seed_centroid,
            axis=1,
        ) <= KEYPOINT_FRUIT_CLUSTER_RADIUS_M

        same_blob = labeled[
            ys,
            xs,
        ] == label

        keep = nearby & same_blob

        if int(
            keep.sum()
        ) < KEYPOINT_MIN_FRUIT_PIXELS:
            keep = member

        mask = np.zeros_like(
            tomato_valid
        )

        mask[
            ys[keep],
            xs[keep],
        ] = True

        candidates.append(
            (
                mask,
                points_world[
                    keep
                ].mean(
                    axis=0
                ),
                score,
            )
        )

    candidates.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    return candidates[
        :max_candidates
    ]


def _target_fruit_cluster(
    tomato_valid,
    depth,
    intrinsics,
    camera_pose,
):
    """
    Best single fruit cluster near the image centre.
    """
    candidates = _fruit_cluster_candidates(
        tomato_valid,
        depth,
        intrinsics,
        camera_pose,
        max_candidates=1,
    )

    if not candidates:
        return None, None

    return candidates[0][0], candidates[0][1]


def _peduncle_points_for_fruit(
    fruit,
    tomato_world,
    peduncle,
    valid,
    depth,
    camera_pose,
):
    """
    Peduncle candidate points attached to one fruit cluster.
    """
    fruit_depths = depth[
        fruit
    ]

    if fruit_depths.size == 0:
        return None, None, None

    depth_low = float(
        np.percentile(
            fruit_depths,
            10,
        )
    ) - KEYPOINT_DEPTH_BAND_M

    depth_high = float(
        np.percentile(
            fruit_depths,
            90,
        )
    ) + KEYPOINT_DEPTH_BAND_M

    candidate = _component_touching(
        _close_gaps(
            peduncle
            & valid
            & (depth >= depth_low)
            & (depth <= depth_high)
        ),
        _dilate(
            fruit,
            KEYPOINT_TOMATO_DILATION,
        ),
    )

    ys, xs = np.where(
        candidate
    )

    if len(
        xs
    ) < KEYPOINT_MIN_PEDUNCLE_PIXELS:
        return None, None, None

    depths = depth[
        ys,
        xs,
    ]

    keep = np.abs(
        depths
        - float(
            np.median(
                depths
            )
        )
    ) <= KEYPOINT_DEPTH_SPECKLE_M

    if int(
        keep.sum()
    ) >= KEYPOINT_MIN_PEDUNCLE_PIXELS:
        xs = xs[keep]
        ys = ys[keep]
        depths = depths[keep]

    points_world = camera_to_world(
        backproject_pixels(
            xs,
            ys,
            depths,
            camera_pose[
                "intrinsics"
            ],
        ),
        camera_pose,
    )

    within_reach = np.linalg.norm(
        points_world - tomato_world,
        axis=1,
    ) <= KEYPOINT_MAX_REACH_M

    if int(
        within_reach.sum()
    ) >= KEYPOINT_MIN_PEDUNCLE_PIXELS:
        xs = xs[within_reach]
        ys = ys[within_reach]
        points_world = points_world[
            within_reach
        ]

    return xs, ys, points_world


def _score_tracked_peduncle(
    points_world,
    tomato_world,
    direction,
):
    """
    How peduncle-like a tracked segment is for this fruit.
    """
    if abs(
        float(
            direction[2]
        )
    ) > KEYPOINT_MAX_VERTICAL_DOT:
        return -1.0, {}

    distances = np.linalg.norm(
        points_world - tomato_world,
        axis=1,
    )

    fruit_end_distance = float(
        distances.min()
    )

    stem_end_distance = float(
        distances.max()
    )

    # Approximate length from robust ends along the axis.
    centroid = points_world.mean(
        axis=0
    )

    projections = (
        points_world - centroid
    ) @ direction

    length = float(
        np.percentile(
            projections,
            100.0
            - KEYPOINT_EXTREME_PERCENTILE,
        )
        - np.percentile(
            projections,
            KEYPOINT_EXTREME_PERCENTILE,
        )
    )

    length = abs(
        length
    )

    if not (
        KEYPOINT_LENGTH_MIN_M
        <= length
        <= KEYPOINT_LENGTH_MAX_M
    ):
        return -1.0, {
            "peduncle_length_m": length,
        }

    if fruit_end_distance > KEYPOINT_ATTACHMENT_MAX_M:
        return -1.0, {
            "peduncle_length_m": length,
        }

    if stem_end_distance < KEYPOINT_MIN_STEM_TO_FRUIT_M:
        return -1.0, {
            "peduncle_length_m": length,
        }

    length_score = float(
        np.exp(
            -abs(
                length
                - KEYPOINT_LENGTH_TARGET_M
            )
            / 0.04
        )
    )

    attachment_score = float(
        np.clip(
            1.0
            - fruit_end_distance
            / KEYPOINT_ATTACHMENT_MAX_M,
            0.0,
            1.0,
        )
    )

    span_score = float(
        np.clip(
            (
                stem_end_distance
                - KEYPOINT_MIN_STEM_TO_FRUIT_M
            )
            / 0.08,
            0.0,
            1.0,
        )
    )

    score = (
        3.0 * length_score
        + 2.0 * attachment_score
        + 2.0 * span_score
    )

    return score, {
        "peduncle_length_m": length,
        "stem_to_fruit_m": stem_end_distance,
        "fruit_end_distance_m": fruit_end_distance,
    }


def _contiguous_segment(
    projections,
    anchor_index,
):
    """
    The unbroken run along the axis that contains `anchor_index`.

    Peduncles from neighbouring trusses lie almost parallel and land
    in the same blob. They are separated by a gap along the axis, so
    keeping only the run that touches this truss's fruit stops K1 from
    jumping onto the neighbour and inflating the peduncle length.
    """
    order = np.argsort(
        projections
    )

    sorted_t = projections[
        order
    ]

    breaks = np.where(
        np.diff(
            sorted_t
        )
        > KEYPOINT_SEGMENT_GAP_M
    )[0]

    starts = np.concatenate(
        (
            [0],
            breaks + 1,
        )
    )

    ends = np.concatenate(
        (
            breaks + 1,
            [len(sorted_t)],
        )
    )

    anchor_rank = int(
        np.where(
            order == anchor_index
        )[0][0]
    )

    for start, end in zip(
        starts,
        ends,
    ):
        if start <= anchor_rank < end:
            keep = np.zeros(
                len(
                    projections
                ),
                dtype=bool,
            )

            keep[
                order[
                    start:end
                ]
            ] = True

            return keep

    return np.ones(
        len(
            projections
        ),
        dtype=bool,
    )


def _select_tube(
    points,
    origin,
    direction,
    fruit_centroid,
):
    """
    Points inside a thin tube about a line, as one run from the fruit.

    The contiguous-run test is what rejects a peduncle crossing this
    one: it shares only the pixels at the intersection, so it sits
    beyond a gap along the axis and is dropped.
    """
    offset = points - origin

    along = offset @ direction

    across = np.linalg.norm(
        offset
        - np.outer(
            along,
            direction,
        ),
        axis=1,
    )

    inside = across <= KEYPOINT_TUBE_RADIUS_M

    if int(
        inside.sum()
    ) < KEYPOINT_MIN_PEDUNCLE_PIXELS:
        return None, None

    inside_index = np.where(
        inside
    )[0]

    run = _contiguous_segment(
        along[
            inside
        ],
        int(
            np.argmin(
                np.linalg.norm(
                    points[
                        inside
                    ]
                    - fruit_centroid,
                    axis=1,
                )
            )
        ),
    )

    keep = np.zeros(
        len(
            points
        ),
        dtype=bool,
    )

    keep[
        inside_index[
            run
        ]
    ] = True

    return keep, along


def _ransac_peduncle(
    points,
    fruit_centroid,
):
    """
    Pick this truss's peduncle out of a thicket of identical ones.

    Pixel count alone prefers thick stems and leaf plates. A real
    peduncle is a roughly horizontal bar of authored length that
    reaches the fruit at one end and sits a peduncle-length away
    from it at the other.
    """
    rng = np.random.default_rng(
        0
    )

    up = np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    if len(
        points
    ) > KEYPOINT_RANSAC_MAX_POINTS:
        sample = points[
            rng.choice(
                len(
                    points
                ),
                KEYPOINT_RANSAC_MAX_POINTS,
                replace=False,
            )
        ]
    else:
        sample = points

    best_keep = None
    best_direction = None
    best_score = -1.0

    for _ in range(
        KEYPOINT_RANSAC_ITERATIONS
    ):
        first, second = rng.integers(
            0,
            len(
                sample
            ),
            size=2,
        )

        span = sample[second] - sample[first]

        length = float(
            np.linalg.norm(
                span
            )
        )

        if not (
            KEYPOINT_LENGTH_MIN_M
            <= length
            <= KEYPOINT_TRACK_MAX_LENGTH_M
        ):
            continue

        direction = span / length

        # Main stem / leaf plate are nearly vertical.
        if abs(
            float(
                np.dot(
                    direction,
                    up,
                )
            )
        ) > KEYPOINT_MAX_VERTICAL_DOT:
            continue

        keep, along = _select_tube(
            sample,
            sample[first],
            direction,
            fruit_centroid,
        )

        if keep is None:
            continue

        kept = sample[
            keep
        ]

        kept_along = along[
            keep
        ]

        segment_length = float(
            kept_along.max()
            - kept_along.min()
        )

        if not (
            KEYPOINT_LENGTH_MIN_M
            <= segment_length
            <= KEYPOINT_TRACK_MAX_LENGTH_M
        ):
            continue

        distances = np.linalg.norm(
            kept - fruit_centroid,
            axis=1,
        )

        fruit_end_distance = float(
            distances.min()
        )

        stem_end_distance = float(
            distances.max()
        )

        # Must reach the fruit, and the other end must look like a
        # stem junction rather than a tomato-side stub.
        if fruit_end_distance > KEYPOINT_ATTACHMENT_MAX_M:
            continue

        if stem_end_distance < KEYPOINT_MIN_STEM_TO_FRUIT_M:
            continue

        length_score = float(
            np.exp(
                -abs(
                    segment_length
                    - KEYPOINT_LENGTH_TARGET_M
                )
                / 0.04
            )
        )

        attachment_score = float(
            np.clip(
                1.0
                - fruit_end_distance
                / KEYPOINT_ATTACHMENT_MAX_M,
                0.0,
                1.0,
            )
        )

        span_score = float(
            np.clip(
                (
                    stem_end_distance
                    - KEYPOINT_MIN_STEM_TO_FRUIT_M
                )
                / 0.08,
                0.0,
                1.0,
            )
        )

        # Mild preference for support, without letting fat stems win.
        support_score = float(
            np.clip(
                keep.sum() / 400.0,
                0.0,
                1.0,
            )
        )

        score = (
            3.0 * length_score
            + 2.0 * attachment_score
            + 2.0 * span_score
            + 0.5 * support_score
        )

        if score > best_score:
            best_score = score
            best_keep = kept
            best_direction = direction

    if best_keep is None:
        return None, None

    try:
        _, refined, _ = fit_line_3d(
            best_keep
        )
    except ValueError:
        return (
            best_keep.mean(
                axis=0
            ),
            best_direction,
        )

    if float(
        np.dot(
            refined,
            best_direction,
        )
    ) < 0.0:
        refined = -refined

    return (
        best_keep.mean(
            axis=0
        ),
        refined,
    )


def detect_truss_keypoints(
    rgb,
    depth,
    camera_pose,
    tool_spacing_m,
):
    """
    Find K1 (CutPoint) and K2 (GraspPoint) on the peduncle.

    The peduncle attached to this truss's fruit is recovered first.
    K1 and K2 are then placed on that axis at the authored cut and
    grasp offsets from the stem end, and each is depth-lifted.
    """
    rgb01 = _as_rgb01(
        rgb
    )

    valid, depth = valid_depth_mask(
        depth
    )

    tomato = tomato_mask(
        rgb01
    )

    peduncle = peduncle_mask(
        rgb01
    )

    diagnostics = {
        "tomato_pixels": int(
            tomato.sum()
        ),
        "peduncle_pixels": int(
            peduncle.sum()
        ),
    }

    # A wrist camera pressed into geometry returns a frame that is all
    # near-field: worth its own reason, since no amount of estimator
    # tuning fixes it but a different viewpoint does.
    if not np.any(
        valid
    ):
        return (
            None,
            "camera_blocked_or_no_depth",
            diagnostics,
        )

    if diagnostics[
        "tomato_pixels"
    ] < KEYPOINT_MIN_TOMATO_PIXELS:
        return None, "no_tomatoes", diagnostics

    intrinsics = camera_pose[
        "intrinsics"
    ]

    # Several trusses share a frame. Try the best few fruit blobs near
    # the aimed centre and keep the peduncle that scores best, so a
    # larger neighbour cannot permanently steal the seed.
    fruit_candidates = _fruit_cluster_candidates(
        tomato & valid,
        depth,
        intrinsics,
        camera_pose,
    )

    if not fruit_candidates:
        return None, "no_tomato_depth", diagnostics

    best = None

    for fruit, tomato_world, fruit_score in fruit_candidates:
        xs, ys, points_world = _peduncle_points_for_fruit(
            fruit,
            tomato_world,
            peduncle,
            valid,
            depth,
            camera_pose,
        )

        if points_world is None:
            continue

        origin, direction = _ransac_peduncle(
            points_world,
            tomato_world,
        )

        if direction is None:
            continue

        tracked, _ = _select_tube(
            points_world,
            origin,
            direction,
            tomato_world,
        )

        if (
            tracked is None
            or int(
                tracked.sum()
            ) < KEYPOINT_MIN_PEDUNCLE_PIXELS
        ):
            continue

        tracked_points = points_world[
            tracked
        ]

        peduncle_score, peduncle_info = _score_tracked_peduncle(
            tracked_points,
            tomato_world,
            direction,
        )

        if peduncle_score < 0.0:
            continue

        # Mild preference for the aimed fruit, without letting it
        # override a clearly better peduncle on another blob.
        total = peduncle_score + 0.15 * float(
            np.clip(
                fruit_score / 5000.0,
                0.0,
                1.0,
            )
        )

        if best is None or total > best["total"]:
            best = {
                "total": total,
                "fruit": fruit,
                "tomato_world": tomato_world,
                "xs": xs[tracked],
                "ys": ys[tracked],
                "points_world": tracked_points,
                "direction": direction,
                "peduncle_info": peduncle_info,
                "fruit_pixels": int(
                    fruit.sum()
                ),
                "candidate_pixels": int(
                    len(
                        xs
                    )
                ),
            }

    if best is None:
        return (
            None,
            "no_peduncle_of_expected_length",
            diagnostics,
        )

    fruit = best["fruit"]
    tomato_world = best["tomato_world"]
    xs = best["xs"]
    ys = best["ys"]
    points_world = best["points_world"]

    diagnostics["fruit_pixels"] = best["fruit_pixels"]
    diagnostics["candidate_pixels"] = best["candidate_pixels"]
    diagnostics["fruit_candidates_tried"] = len(
        fruit_candidates
    )

    try:
        centroid, axis, residual_m = fit_line_3d(
            points_world
        )
    except ValueError:
        return (
            None,
            "degenerate_peduncle_line",
            diagnostics,
        )

    if abs(
        float(
            axis[2]
        )
    ) > KEYPOINT_MAX_VERTICAL_DOT:
        return (
            None,
            "peduncle_too_vertical",
            diagnostics,
        )

    projections = (
        points_world - centroid
    ) @ axis

    t_low = float(
        np.percentile(
            projections,
            KEYPOINT_EXTREME_PERCENTILE,
        )
    )

    t_high = float(
        np.percentile(
            projections,
            100.0
            - KEYPOINT_EXTREME_PERCENTILE,
        )
    )

    end_low = centroid + axis * t_low
    end_high = centroid + axis * t_high

    # Stem end is farther from the fruit; fruit end is closer.
    if float(
        np.linalg.norm(
            end_low - tomato_world
        )
    ) >= float(
        np.linalg.norm(
            end_high - tomato_world
        )
    ):
        stem_world = end_low
        fruit_end = end_high
    else:
        stem_world = end_high
        fruit_end = end_low

    direction = fruit_end - stem_world

    direction_norm = float(
        np.linalg.norm(
            direction
        )
    )

    if direction_norm <= 1.0e-6:
        return (
            None,
            "degenerate_peduncle_line",
            diagnostics,
        )

    direction = direction / direction_norm

    peduncle_length = direction_norm

    diagnostics["line_residual_m"] = float(
        residual_m
    )

    diagnostics["segment_pixels"] = int(
        len(
            xs
        )
    )

    diagnostics["peduncle_length_m"] = float(
        peduncle_length
    )

    diagnostics["stem_to_fruit_m"] = float(
        np.linalg.norm(
            stem_world - tomato_world
        )
    )

    if not (
        KEYPOINT_LENGTH_MIN_M
        <= peduncle_length
        <= KEYPOINT_LENGTH_MAX_M
    ):
        return (
            None,
            f"peduncle_length_out_of_range ({peduncle_length:.3f} m)",
            diagnostics,
        )

    if diagnostics[
        "stem_to_fruit_m"
    ] < KEYPOINT_MIN_STEM_TO_FRUIT_M:
        return (
            None,
            "stem_end_too_close_to_fruit",
            diagnostics,
        )

    # Recompute projections on the stem -> fruit axis for support.
    projections = (
        points_world - centroid
    ) @ direction

    # Task points, not peduncle tips.
    targets = {
        "k1": (
            stem_world
            + direction
            * KEYPOINT_CUT_OFFSET_M
        ),
        "k2": (
            stem_world
            + direction
            * (
                KEYPOINT_CUT_OFFSET_M
                + float(
                    tool_spacing_m
                )
            )
        ),
    }

    keypoints = {}

    for name, target_world in targets.items():
        t_target = float(
            np.dot(
                target_world - centroid,
                direction,
            )
        )

        index = int(
            np.argmin(
                np.abs(
                    projections
                    - t_target
                )
            )
        )

        u = float(
            xs[index]
        )
        v = float(
            ys[index]
        )

        # Prefer the camera projection of the axis point when it
        # lands on the image; fall back to the nearest segment pixel.
        projected = project_world_to_pixel(
            target_world,
            camera_pose,
        )

        if projected is not None:
            height, width = rgb01.shape[
                :2
            ]

            if (
                0.0 <= projected[0] < width
                and 0.0 <= projected[1] < height
            ):
                u = float(
                    projected[0]
                )
                v = float(
                    projected[1]
                )

        patch_depth, depth_fraction = _patch_depth(
            depth,
            valid,
            u,
            v,
        )

        if patch_depth is None:
            # Camera projection can land just off the peduncle; the
            # nearest segment pixel still gives a usable depth lift.
            u = float(
                xs[index]
            )
            v = float(
                ys[index]
            )

            patch_depth, depth_fraction = _patch_depth(
                depth,
                valid,
                u,
                v,
            )

        if patch_depth is None:
            return (
                None,
                f"{name}_no_depth",
                diagnostics,
            )

        lifted = camera_to_world(
            backproject_pixels(
                [u],
                [v],
                [patch_depth],
                intrinsics,
            ),
            camera_pose,
        )[0]

        offset_vector = (
            lifted - centroid
        )

        line_offset_m = float(
            np.linalg.norm(
                offset_vector
                - float(
                    np.dot(
                        offset_vector,
                        direction,
                    )
                )
                * direction
            )
        )

        support_pixels = int(
            np.sum(
                np.abs(
                    projections
                    - t_target
                )
                <= KEYPOINT_SUPPORT_WINDOW_M
            )
        )

        border = _border_score(
            u,
            v,
            rgb01.shape,
        )

        confidence = _keypoint_confidence(
            depth_valid_fraction=depth_fraction,
            support_pixels=support_pixels,
            border_score=border,
            line_offset_m=line_offset_m,
        )

        # Keep the authored position along the peduncle; depth only
        # corrects the across-axis drift of the lift.
        snapped = (
            centroid
            + direction * t_target
        )

        keypoints[name] = {
            "pixel": np.array(
                [u, v],
                dtype=np.float64,
            ),
            "world": np.asarray(
                snapped,
                dtype=np.float64,
            ),
            "world_raw": np.asarray(
                lifted,
                dtype=np.float64,
            ),
            "depth_m": float(
                patch_depth
            ),
            "depth_valid_fraction": float(
                depth_fraction
            ),
            "support_pixels": support_pixels,
            "border_score": border,
            "line_offset_m": line_offset_m,
            "confidence": confidence,
            "role": (
                "cut"
                if name == "k1"
                else "grasp"
            ),
        }

    keypoints["axis_direction"] = np.asarray(
        direction,
        dtype=np.float64,
    )

    # Pixels the keypoints were read from, for overlays.
    keypoints["segment_uv"] = np.stack(
        (
            xs,
            ys,
        ),
        axis=1,
    )

    keypoints["line_residual_m"] = float(
        residual_m
    )

    return keypoints, None, diagnostics


def derive_targets_from_keypoints(
    k1_world,
    k2_world,
    tool_spacing_m=None,
    derivation=None,
):
    """
    CutPoint / GraspPoint are the keypoints themselves.

    K1 = CutPoint, K2 = GraspPoint.
    `tool_spacing_m` / `derivation` are accepted for call-site
    compatibility and ignored.
    """
    _ = tool_spacing_m, derivation

    k1 = np.asarray(
        k1_world,
        dtype=np.float64,
    ).reshape(3)

    k2 = np.asarray(
        k2_world,
        dtype=np.float64,
    ).reshape(3)

    span = k2 - k1

    length = float(
        np.linalg.norm(
            span
        )
    )

    if length <= 1.0e-6:
        raise ValueError(
            "K1 and K2 coincide; no peduncle direction."
        )

    direction = span / length

    return (
        k2,
        k1,
        direction,
        length,
    )


def estimate_keypoint_targets(
    rgb,
    depth,
    camera_pose,
    tool_spacing_m,
    derivation=None,
):
    """
    Full keypoint estimate for one RGB-D view.
    """
    _ = derivation

    if (
        rgb is None
        or depth is None
        or camera_pose is None
    ):
        return _failure(
            "missing_rgb_depth_or_pose",
            tool_spacing_m=tool_spacing_m,
        )

    keypoints, reason, diagnostics = (
        detect_truss_keypoints(
            rgb,
            depth,
            camera_pose,
            tool_spacing_m=tool_spacing_m,
        )
    )

    if keypoints is None:
        return _failure(
            reason,
            diagnostics,
            tool_spacing_m,
        )

    k1 = keypoints["k1"]
    k2 = keypoints["k2"]

    try:
        grasp, cut, direction, spacing = (
            derive_targets_from_keypoints(
                k1["world"],
                k2["world"],
                tool_spacing_m,
            )
        )
    except ValueError:
        return _failure(
            "k1_k2_coincident",
            diagnostics,
            tool_spacing_m,
        )

    diagnostics["k1_k2_spacing_m"] = spacing

    spacing_ok = abs(
        spacing
        - float(
            tool_spacing_m
        )
    ) <= KEYPOINT_SPACING_TOL_M

    confidence = min(
        k1["confidence"],
        k2["confidence"],
    )

    if not spacing_ok:
        confidence *= KEYPOINT_BAD_SPACING_PENALTY

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )

    needs_another_view = bool(
        confidence < KEYPOINT_MIN_CONFIDENCE
        or not spacing_ok
        or k1["confidence"]
        < KEYPOINT_MIN_SINGLE_CONFIDENCE
        or k2["confidence"]
        < KEYPOINT_MIN_SINGLE_CONFIDENCE
    )

    reasons = []

    if not spacing_ok:
        reasons.append(
            f"k1_k2_spacing_mismatch "
            f"({spacing:.3f} m vs {float(tool_spacing_m):.3f} m)"
        )

    if (
        k1["confidence"]
        < KEYPOINT_MIN_SINGLE_CONFIDENCE
    ):
        reasons.append(
            "k1_low_confidence"
        )

    if (
        k2["confidence"]
        < KEYPOINT_MIN_SINGLE_CONFIDENCE
    ):
        reasons.append(
            "k2_low_confidence"
        )

    return {
        "grasp_point_world": np.asarray(
            grasp,
            dtype=np.float64,
        ),
        "cut_point_world": np.asarray(
            cut,
            dtype=np.float64,
        ),
        "peduncle_direction": np.asarray(
            direction,
            dtype=np.float64,
        ),
        "confidence": confidence,
        "source": "keypoints",
        "views_used": 1,
        "tool_spacing_m": float(
            tool_spacing_m
        ),
        "derivation": "k1_cut_k2_grasp",
        "keypoints": keypoints,
        "needs_another_view": needs_another_view,
        "reason": (
            "; ".join(
                reasons
            )
            if reasons
            else ""
        ),
        "diagnostics": diagnostics,
    }


class KeypointTargetEstimator:
    """
    Fixed-view keypoint estimator behind estimate_target(...).

    One view per call. The returned `needs_another_view` flag is the
    hook the active-perception condition will act on; the fixed-view
    condition records it and harvests (or skips) on this one view.
    """

    def __init__(
        self,
        stage=None,
        tool_spacing_m=None,
        derivation=None,
    ):
        _ = stage, derivation

        if (
            tool_spacing_m is None
            or float(
                tool_spacing_m
            ) <= 0.0
        ):
            raise ValueError(
                "KeypointTargetEstimator requires a positive "
                "tool_spacing_m."
            )

        self.tool_spacing_m = float(
            tool_spacing_m
        )

    @property
    def name(self):
        return "keypoints"

    def estimate_target(
        self,
        rgb,
        depth,
        camera_pose,
    ):
        return estimate_keypoint_targets(
            rgb=rgb,
            depth=depth,
            camera_pose=camera_pose,
            tool_spacing_m=self.tool_spacing_m,
        )


_SEGMENT_COLOR = (60, 140, 255)
_LIVE_PREVIEW = None


def _draw_cross(
    draw,
    pixel,
    color,
    label,
    radius=8,
    width=2,
):
    if pixel is None:
        return

    u = float(
        pixel[0]
    )
    v = float(
        pixel[1]
    )

    draw.line(
        [
            (u - radius, v),
            (u + radius, v),
        ],
        fill=color,
        width=width,
    )

    draw.line(
        [
            (u, v - radius),
            (u, v + radius),
        ],
        fill=color,
        width=width,
    )

    if label:
        draw.text(
            (
                u + radius + 2,
                v - radius,
            ),
            label,
            fill=color,
        )


def render_keypoint_overlay(
    rgb,
    estimate,
    camera_pose,
    *,
    label="",
    gt_k1_world=None,
    gt_k2_world=None,
    gt_grasp_world=None,
    gt_cut_world=None,
):
    """
    RGB with the keypoints and derived targets drawn on it.

    Used both live (one frame per truss, as soon as it is estimated)
    and offline (replay a finished run).
    """
    from PIL import Image, ImageDraw

    rgb01 = _as_rgb01(
        rgb
    )

    image = Image.fromarray(
        np.clip(
            rgb01 * 255.0,
            0,
            255,
        ).astype(
            np.uint8
        ),
        mode="RGB",
    )

    draw = ImageDraw.Draw(
        image
    )

    keypoints = (
        estimate.get(
            "keypoints"
        )
        or {}
    )

    reason = estimate.get(
        "reason"
    ) or estimate.get(
        "diagnostics",
        {},
    ).get(
        "reason",
        "",
    )

    if not keypoints.get(
        "k1"
    ):
        draw.text(
            (6, 6),
            f"{label}  FAIL {reason}".strip(),
            fill=(255, 80, 80),
        )

        return image

    if keypoints.get(
        "segment_uv"
    ) is not None:
        pixels = image.load()

        height, width = rgb01.shape[
            :2
        ]

        for u, v in keypoints[
            "segment_uv"
        ]:
            iu = int(
                u
            )
            iv = int(
                v
            )

            if (
                0 <= iu < width
                and 0 <= iv < height
            ):
                pixels[
                    iu,
                    iv,
                ] = _SEGMENT_COLOR

    _draw_cross(
        draw,
        keypoints["k1"]["pixel"],
        (0, 255, 255),
        f"K1/cut {keypoints['k1']['confidence']:.2f}",
    )

    _draw_cross(
        draw,
        keypoints["k2"]["pixel"],
        (255, 0, 255),
        f"K2/grasp {keypoints['k2']['confidence']:.2f}",
    )

    # GT for the same two task points. Prefer explicit cut/grasp args;
    # fall back to gt_k1 / gt_k2 when callers only pass those.
    if gt_cut_world is None:
        gt_cut_world = gt_k1_world

    if gt_grasp_world is None:
        gt_grasp_world = gt_k2_world

    if camera_pose is not None:
        for point, color, name in (
            (
                gt_cut_world,
                (160, 0, 0),
                "GT cut",
            ),
            (
                gt_grasp_world,
                (0, 160, 0),
                "GT grasp",
            ),
        ):
            if point is None:
                continue

            _draw_cross(
                draw,
                project_world_to_pixel(
                    point,
                    camera_pose,
                ),
                color,
                name,
                radius=5,
                width=1,
            )

    length = (
        estimate.get(
            "diagnostics",
            {},
        )
        or {}
    ).get(
        "peduncle_length_m"
    )

    spacing = (
        estimate.get(
            "diagnostics",
            {},
        )
        or {}
    ).get(
        "k1_k2_spacing_m"
    )

    caption = label

    if spacing is not None:
        caption = (
            f"{label}  K1-K2 {float(spacing) * 100:.1f} cm"
        ).strip()
    elif length is not None:
        caption = (
            f"{label}  peduncle {float(length) * 100:.1f} cm"
        ).strip()

    if estimate.get(
        "needs_another_view"
    ):
        caption += "  [another view]"

    draw.text(
        (6, 6),
        caption,
        fill=(255, 255, 0),
    )

    return image


def write_keypoint_overlay(
    output_dir,
    image_stem,
    rgb,
    estimate,
    camera_pose,
    **kwargs,
):
    """
    Save one overlay PNG. Safe to call during the route: the file
    appears as soon as that truss has been estimated.
    """
    from pathlib import Path

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / f"{image_stem}.png"

    render_keypoint_overlay(
        rgb,
        estimate,
        camera_pose,
        **kwargs,
    ).save(
        path
    )

    return path


def show_keypoint_overlay(
    image,
    caption="",
):
    """
    Refresh the Isaac 'Keypoint detection' window with this frame.

    No-op outside Isaac. The window is created once and then reused
    so each new truss replaces the previous picture. A UI failure
    must never abort the harvest: the PNG overlay is the durable
    record.
    """
    if not SHOW_KEYPOINT_OVERLAY:
        return

    global _LIVE_PREVIEW

    try:
        import omni.ui as ui
    except ImportError:
        return

    try:
        rgb = np.asarray(
            image.convert(
                "RGB"
            )
            if hasattr(
                image,
                "convert",
            )
            else image,
            dtype=np.uint8,
        )

        if rgb.ndim != 3:
            return

        height, width = rgb.shape[
            :2
        ]

        rgba = np.dstack(
            (
                rgb,
                np.full(
                    (height, width),
                    255,
                    dtype=np.uint8,
                ),
            )
        )

        if _LIVE_PREVIEW is None:
            window = ui.Window(
                "Keypoint detection",
                width=width + 24,
                height=height + 56,
            )

            with window.frame:
                with ui.VStack(
                    spacing=6,
                ):
                    title = ui.Label(
                        caption
                        or "Waiting for first estimate...",
                    )

                    provider = ui.ByteImageProvider()

                    # Isaac's ImageWithProvider takes IwpFillPolicy,
                    # not the plain FillPolicy used by ui.Image.
                    ui.ImageWithProvider(
                        provider,
                        width=width,
                        height=height,
                        fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                    )

            _LIVE_PREVIEW = {
                "window": window,
                "title": title,
                "provider": provider,
            }

        _LIVE_PREVIEW[
            "title"
        ].text = caption

        _LIVE_PREVIEW[
            "provider"
        ].set_bytes_data(
            rgba.flatten().tolist(),
            [
                width,
                height,
            ],
        )

        _LIVE_PREVIEW[
            "window"
        ].visible = True
    except Exception as exc:
        print(
            f"[OVERLAY] live window failed: {exc}"
        )
