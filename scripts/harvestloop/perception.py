"""Wrist camera setup and RGB viewpoint capture."""

from pathlib import Path

from PIL import Image
import numpy as np
from pxr import Gf, UsdGeom

import omni.kit.app
import omni.replicator.core as rep

from usd_utils import get_world_pose

from .config import *  # noqa: F401,F403
from .mathutils import look_rotation_from_points
from .perf import set_render_product_updates
from .usd_pose import world_point_to_frame


def ensure_wrist_camera(
    stage,
):
    """
    Prefer the permanent WristCamera. If missing, create it under
    the gripper base_link using the documented local mount.
    """
    camera_prim = stage.GetPrimAtPath(
        WRIST_CAMERA_PATH
    )

    if camera_prim.IsValid():
        print(
            f"[WRIST CAMERA] Found: {WRIST_CAMERA_PATH}"
        )
        return WRIST_CAMERA_PATH

    gripper_prim = stage.GetPrimAtPath(
        GRIPPER_BASE_PATH
    )

    if not gripper_prim.IsValid():
        raise RuntimeError(
            "Cannot create WristCamera; missing gripper base_link: "
            f"{GRIPPER_BASE_PATH}"
        )

    print()
    print("========================================")
    print("[WRIST CAMERA] CREATING DOCUMENTED MOUNT")
    print("========================================")

    print(
        f"Path: {WRIST_CAMERA_PATH}"
    )

    print(
        "Local position: "
        f"{np.round(WRIST_CAMERA_LOCAL_POSITION, 4)}"
    )

    print(
        "Local look target: "
        f"{np.round(WRIST_CAMERA_LOCAL_LOOK_TARGET, 4)}"
    )

    camera = UsdGeom.Camera.Define(
        stage,
        WRIST_CAMERA_PATH,
    )

    camera.CreateFocalLengthAttr(12.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.2908)
    camera.CreateClippingRangeAttr(
        Gf.Vec2f(0.01, 10.0)
    )

    xformable = UsdGeom.Xformable(
        camera.GetPrim()
    )

    xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp()
    translate_op.Set(
        Gf.Vec3d(
            float(WRIST_CAMERA_LOCAL_POSITION[0]),
            float(WRIST_CAMERA_LOCAL_POSITION[1]),
            float(WRIST_CAMERA_LOCAL_POSITION[2]),
        )
    )

    orient_op = xformable.AddOrientOp()
    orient_op.Set(
        look_rotation_from_points(
            WRIST_CAMERA_LOCAL_POSITION,
            WRIST_CAMERA_LOCAL_LOOK_TARGET,
        )
    )

    return WRIST_CAMERA_PATH


def wrist_camera_intrinsics(
    stage,
    camera_path,
):
    """
    Focal length and apertures as authored on the camera prim.

    Read from USD rather than assumed, because the permanent
    WristCamera in harvest_bot.usd may differ from the fallback
    mount created here.
    """
    camera = UsdGeom.Camera(
        stage.GetPrimAtPath(
            camera_path
        )
    )

    if not camera:
        raise RuntimeError(
            f"Not a camera prim: {camera_path}"
        )

    width, height = WRIST_CAPTURE_RESOLUTION

    return {
        "focal_length": float(
            camera.GetFocalLengthAttr().Get()
        ),
        "horizontal_aperture": float(
            camera.GetHorizontalApertureAttr().Get()
        ),
        "vertical_aperture": float(
            camera.GetVerticalApertureAttr().Get()
        ),
        "width": int(width),
        "height": int(height),
    }


def project_world_point_to_wrist_image(
    point_in_camera,
    intrinsics,
):
    """
    Pixel coordinates of a point already expressed in camera frame.

    USD cameras look down local -Z with +X right and +Y up. The
    render product fits the HORIZONTAL aperture and derives the
    vertical extent from the image aspect ratio, which is why the
    authored vertical aperture is not used here (640x480 is not the
    same aspect as the authored 20.955 x 15.291 film back).

    Returns None when the point is behind the camera.
    """
    depth = -float(
        point_in_camera[2]
    )

    if depth <= 1.0e-6:
        return None

    width = intrinsics["width"]
    height = intrinsics["height"]

    half_aperture = (
        0.5
        * intrinsics["horizontal_aperture"]
    )

    scale = (
        intrinsics["focal_length"]
        / depth
    )

    x_ndc = (
        scale
        * float(point_in_camera[0])
        / half_aperture
    )

    y_ndc = (
        scale
        * float(point_in_camera[1])
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


def wrist_camera_framing(
    stage,
    world_point,
    camera_path=None,
):
    """
    Where a world point currently falls in the wrist image.

    Used to close the loop on aiming: the viewpoint stage keeps
    correcting until the truss is inside the frame, measured rather
    than assumed.

    Returns a dict with the pixel position, its distance from the
    image center as a fraction of the half-frame, and whether the
    point is inside the central region.
    """
    camera_path = (
        camera_path
        or WRIST_CAMERA_PATH
    )

    intrinsics = wrist_camera_intrinsics(
        stage,
        camera_path,
    )

    point_in_camera = world_point_to_frame(
        stage,
        camera_path,
        world_point,
    )

    pixel = project_world_point_to_wrist_image(
        point_in_camera,
        intrinsics,
    )

    if pixel is None:
        return {
            "pixel": None,
            "depth": -float(
                point_in_camera[2]
            ),
            "center_fraction": float(
                "inf"
            ),
            "in_view": False,
        }

    # Distance from the image center, as a fraction of the half
    # frame: 0 is centered, 1 is the edge.
    center_fraction = max(
        abs(
            pixel[0]
            - 0.5 * intrinsics["width"]
        )
        / (0.5 * intrinsics["width"]),
        abs(
            pixel[1]
            - 0.5 * intrinsics["height"]
        )
        / (0.5 * intrinsics["height"]),
    )

    return {
        "pixel": pixel,
        "depth": -float(
            point_in_camera[2]
        ),
        "center_fraction": float(
            center_fraction
        ),
        "in_view": bool(
            center_fraction <= 1.0
        ),
    }


def label_points_in_wrist_camera(
    stage,
    camera_path,
    label_points,
):
    """
    Privileged world points as camera-frame coordinates and pixels.

    These are the ground-truth labels for the captured image. They
    are measured at capture time, while the arm still holds the
    viewpoint pose.
    """
    intrinsics = wrist_camera_intrinsics(
        stage,
        camera_path,
    )

    labels = {}

    for name, world_point in label_points.items():
        if world_point is None:
            continue

        point_in_camera = world_point_to_frame(
            stage,
            camera_path,
            world_point,
        )

        pixel = project_world_point_to_wrist_image(
            point_in_camera,
            intrinsics,
        )

        in_view = (
            pixel is not None
            and 0.0 <= pixel[0] < intrinsics["width"]
            and 0.0 <= pixel[1] < intrinsics["height"]
        )

        labels[name] = {
            "world": np.asarray(
                world_point,
                dtype=np.float64,
            ),
            "camera": point_in_camera,
            "pixel": pixel,
            "depth": -float(
                point_in_camera[2]
            ),
            "in_view": bool(
                in_view
            ),
        }

    return labels, intrinsics


def _annotator_array(
    value,
):
    if isinstance(
        value,
        dict,
    ):
        if "data" not in value:
            raise RuntimeError(
                "Annotator returned a dict without a 'data' field."
            )

        value = value[
            "data"
        ]

    return np.asarray(
        value
    )


async def setup_persistent_wrist_rgbd_capture(
    stage,
):
    """
    Create the WristCamera RGB-D render product ONCE.

    IMPORTANT:
    Call this BEFORE SingleArticulation.initialize().
    Creating Replicator render-product prims after articulation
    tensor views exist can invalidate those views and break RMPflow.
    """
    camera_path = ensure_wrist_camera(
        stage
    )

    print()
    print("========================================")
    print("[WRIST CAPTURE] PERSISTENT RGB-D SETUP")
    print("========================================")

    print(
        f"Camera: {camera_path}"
    )

    print(
        f"Resolution: "
        f"{WRIST_CAPTURE_RESOLUTION[0]} x "
        f"{WRIST_CAPTURE_RESOLUTION[1]}"
    )

    render_product = (
        rep.create.render_product(
            camera_path,
            WRIST_CAPTURE_RESOLUTION,
        )
    )

    rgb_annot = (
        rep.AnnotatorRegistry
        .get_annotator(
            "rgb"
        )
    )

    depth_annot = (
        rep.AnnotatorRegistry
        .get_annotator(
            "distance_to_image_plane"
        )
    )

    rgb_annot.attach(
        [
            render_product
        ]
    )

    depth_annot.attach(
        [
            render_product
        ]
    )

    # Annotator/render-product warm-up only. No robot commands.
    for _ in range(
        WRIST_CAPTURE_WARMUP_FRAMES
    ):
        await (
            omni.kit.app
            .get_app()
            .next_update_async()
        )

    capture_session = {
        "camera_path": camera_path,
        "render_product": render_product,
        "rgb_annot": rgb_annot,
        "depth_annot": depth_annot,
    }

    # Warm-up is done, so stop the wrist camera rendering every app
    # update for the rest of the route. capture_wrist_rgbd switches
    # it back on around the frames it actually reads.
    capture_session["gated"] = set_render_product_updates(
        capture_session,
        False,
    )

    if capture_session["gated"]:
        print(
            "[PERF] Wrist render product idles between captures."
        )

    return capture_session


# Older call sites / docs.
setup_persistent_wrist_rgb_capture = (
    setup_persistent_wrist_rgbd_capture
)


async def capture_wrist_rgbd(
    stage,
    image_stem,
    capture_session,
    output_dir=None,
    label_points=None,
):
    """
    Passive WristCamera RGB-D capture for the perception seam.

    Returns arrays the estimator consumes, plus on-disk paths and
    optional privileged labels used only for scoring / overlays.

    Does NOT:
        - create/destroy USD prims
        - command Jackal/arm
        - change RMPflow targets
        - pause the timeline
        - call orchestrator.stop_async()
    """
    if capture_session is None:
        raise RuntimeError(
            "Wrist capture session was not set up before "
            "articulation initialization."
        )

    camera_path = capture_session[
        "camera_path"
    ]

    rgb_annot = capture_session[
        "rgb_annot"
    ]

    depth_annot = capture_session.get(
        "depth_annot"
    )

    image_directory = Path(
        output_dir
        if output_dir is not None
        else VIEWPOINT_OUTPUT_DIR
    )

    image_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_path = (
        image_directory
        / f"{image_stem}.png"
    )

    depth_path = (
        image_directory
        / f"{image_stem}_depth.npy"
    )

    camera_position, camera_orientation = (
        get_world_pose(
            stage,
            camera_path,
        )
    )

    labels, intrinsics = label_points_in_wrist_camera(
        stage,
        camera_path,
        label_points or {},
    )

    print()
    print("========================================")
    print(f"[WRIST RGB-D CAPTURE] {image_stem}")
    print("========================================")

    print(
        f"Camera: {camera_path}"
    )

    print(
        "Wrist camera world position: "
        f"{np.round(camera_position, 4)}"
    )

    print(
        "Wrist camera world quat [wxyz]: "
        f"{np.round(camera_orientation, 4)}"
    )

    print(
        f"Image path: {image_path}"
    )

    print(
        f"Depth path: {depth_path}"
    )

    for name, label in labels.items():
        print(
            f"[LABEL] {name}: "
            f"cam={np.round(label['camera'], 4)} | "
            f"depth={label['depth']:+.4f} m | "
            f"pixel="
            f"{np.round(label['pixel'], 1) if label['pixel'] is not None else 'behind'} | "
            f"in_view={label['in_view']}"
        )

    print(
        "[CAPTURE MODE] Passive RGB-D | "
        "delta_time=0.0 | pause_timeline=False | "
        "persistent render product"
    )

    # The wrist camera idles between captures, so switch it on for the
    # render steps that produce this image and off again after.
    gated = set_render_product_updates(
        capture_session,
        True,
    )

    try:
        # A render target that was idling can return the frame it was
        # switched off on, so pay one extra step to flush it before
        # the steps whose result is saved.
        render_steps = (
            WRIST_CAPTURE_RENDER_STEPS
            + (1 if gated else 0)
        )

        for _ in range(
            render_steps
        ):
            await (
                rep.orchestrator
                .step_async(
                    rt_subframes=4,
                    delta_time=0.0,
                    pause_timeline=False,
                )
            )

        rgb = _annotator_array(
            rgb_annot.get_data()
        )

        depth = None

        if depth_annot is not None:
            depth = _annotator_array(
                depth_annot.get_data()
            )
    finally:
        if gated:
            set_render_product_updates(
                capture_session,
                False,
            )

    if rgb.size == 0:
        raise RuntimeError(
            "RGB annotator returned an empty image."
        )

    if (
        rgb.ndim == 3
        and rgb.shape[-1] == 4
    ):
        rgb = rgb[
            :,
            :,
            :3,
        ]

    rgb = np.asarray(
        rgb,
        dtype=np.uint8,
    )

    Image.fromarray(
        rgb
    ).save(
        image_path
    )

    if depth is not None:
        depth = np.asarray(
            depth,
            dtype=np.float32,
        )

        # Some annotators return HxWx1.
        if (
            depth.ndim == 3
            and depth.shape[-1] == 1
        ):
            depth = depth[
                :,
                :,
                0,
            ]

        np.save(
            depth_path,
            depth,
        )

    print(
        f"[CAPTURE OK] saved {image_path}"
        + (
            f" | depth={depth_path.name}"
            if depth is not None
            else " | depth=missing"
        )
    )

    camera_pose = {
        "position": np.asarray(
            camera_position,
            dtype=np.float64,
        ),
        "orientation": np.asarray(
            camera_orientation,
            dtype=np.float64,
        ),
        "intrinsics": intrinsics,
        "camera_path": camera_path,
    }

    return {
        "image_path": str(
            image_path
        ),
        "depth_path": (
            str(depth_path)
            if depth is not None
            else None
        ),
        "rgb": rgb,
        "depth": depth,
        "camera_pose": camera_pose,
        "camera_position": camera_pose[
            "position"
        ],
        "camera_orientation": camera_pose[
            "orientation"
        ],
        "labels": labels,
        "intrinsics": intrinsics,
    }


async def capture_wrist_rgb(
    stage,
    image_stem,
    capture_session,
    output_dir=None,
    label_points=None,
):
    """
    Back-compat wrapper around RGB-D capture.
    """
    return await capture_wrist_rgbd(
        stage,
        image_stem,
        capture_session,
        output_dir=output_dir,
        label_points=label_points,
    )
