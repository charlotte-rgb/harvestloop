"""Primitive geometry creation and small vector helpers."""

import math

from pxr import Gf, UsdGeom


# =========================================================
# BASIC GEOMETRY
# =========================================================

def normalize(v):
    length = v.GetLength()

    if length < 1e-8:
        raise ValueError(
            "Zero-length vector"
        )

    return v / length


def get_world_position(
    stage,
    prim_path,
):
    prim = stage.GetPrimAtPath(
        prim_path
    )

    if not prim.IsValid():
        raise RuntimeError(
            f"Invalid prim path: {prim_path}"
        )

    cache = UsdGeom.XformCache()

    matrix = (
        cache.GetLocalToWorldTransform(
            prim
        )
    )

    return matrix.ExtractTranslation()


def create_box(
    stage,
    path,
    position,
    dimensions,
):
    cube = UsdGeom.Cube.Define(
        stage,
        path,
    )

    cube.CreateSizeAttr(1.0)

    xform = UsdGeom.Xformable(
        cube.GetPrim()
    )

    xform.AddTranslateOp().Set(
        Gf.Vec3d(*position)
    )

    xform.AddScaleOp().Set(
        Gf.Vec3f(*dimensions)
    )

    return cube.GetPrim()


def create_sphere(
    stage,
    path,
    position,
    radius,
):
    sphere = UsdGeom.Sphere.Define(
        stage,
        path,
    )

    sphere.CreateRadiusAttr(
        radius
    )

    UsdGeom.Xformable(
        sphere.GetPrim()
    ).AddTranslateOp().Set(
        Gf.Vec3d(*position)
    )

    return sphere.GetPrim()


def create_cylinder_between(
    stage,
    path,
    p1,
    p2,
    radius,
):
    p1 = Gf.Vec3d(*p1)
    p2 = Gf.Vec3d(*p2)

    direction = p2 - p1
    length = direction.GetLength()

    if length < 1e-6:
        raise ValueError(
            f"{path}: endpoints too close"
        )

    direction = normalize(direction)

    midpoint = (
        p1 + p2
    ) * 0.5

    cylinder = UsdGeom.Cylinder.Define(
        stage,
        path,
    )

    cylinder.CreateRadiusAttr(
        radius
    )

    cylinder.CreateHeightAttr(
        length
    )

    xform = UsdGeom.Xformable(
        cylinder.GetPrim()
    )

    xform.AddTranslateOp().Set(
        midpoint
    )

    rotation = Gf.Rotation(
        Gf.Vec3d(0, 0, 1),
        direction,
    )

    quat = rotation.GetQuat()
    imag = quat.GetImaginary()

    xform.AddOrientOp().Set(
        Gf.Quatf(
            float(quat.GetReal()),
            Gf.Vec3f(
                float(imag[0]),
                float(imag[1]),
                float(imag[2]),
            ),
        )
    )

    return cylinder.GetPrim()


def point_on_segment(
    p1,
    p2,
    t,
):
    a = Gf.Vec3d(*p1)
    b = Gf.Vec3d(*p2)

    return tuple(
        a + (b - a) * t
    )


def generate_direction(
    yaw_deg,
    pitch_deg,
):
    yaw = math.radians(
        yaw_deg
    )

    pitch = math.radians(
        pitch_deg
    )

    return Gf.Vec3d(
        math.cos(pitch)
        * math.cos(yaw),

        math.cos(pitch)
        * math.sin(yaw),

        math.sin(pitch),
    )
