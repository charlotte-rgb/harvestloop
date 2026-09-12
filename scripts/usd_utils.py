import numpy as np
from pxr import Usd, UsdGeom


def get_world_position(stage, path):
    prim = stage.GetPrimAtPath(path)

    T = UsdGeom.Xformable(
        prim
    ).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )

    p = T.ExtractTranslation()

    return np.array(
        [p[0], p[1], p[2]],
        dtype=float,
    )


def get_world_pose(stage, path):
    prim = stage.GetPrimAtPath(path)

    T = UsdGeom.Xformable(
        prim
    ).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )

    p = T.ExtractTranslation()

    rigid = T.RemoveScaleShear()
    q = rigid.ExtractRotationQuat()

    position = np.array(
        [p[0], p[1], p[2]],
        dtype=float,
    )

    imag = q.GetImaginary()

    orientation = np.array(
        [
            q.GetReal(),
            imag[0],
            imag[1],
            imag[2],
        ],
        dtype=float,
    )

    orientation /= np.linalg.norm(
        orientation
    )

    return position, orientation
