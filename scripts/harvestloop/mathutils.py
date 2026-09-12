"""Quaternion / angle helpers (Isaac [w, x, y, z] convention)."""

import math

import numpy as np
from pxr import Gf


# ============================================================
# HELPERS
# ============================================================

def wrap_angle(angle):
    return (
        angle + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def yaw_quaternion(yaw):
    """
    World-Z rotation in Isaac [w, x, y, z] order.
    """
    half = 0.5 * float(yaw)

    return np.array(
        [
            math.cos(half),
            0.0,
            0.0,
            math.sin(half),
        ],
        dtype=np.float64,
    )


def quaternion_to_rpy(q):
    q = np.asarray(
        q,
        dtype=np.float64,
    ).reshape(-1)

    if q.size != 4:
        raise ValueError(
            f"Expected [w,x,y,z], got {q}"
        )

    w, x, y, z = q

    sinr_cosp = 2.0 * (
        w * x + y * z
    )
    cosr_cosp = 1.0 - 2.0 * (
        x * x + y * y
    )
    roll = math.atan2(
        sinr_cosp,
        cosr_cosp,
    )

    sinp = 2.0 * (
        w * y - z * x
    )

    if abs(sinp) >= 1.0:
        pitch = math.copysign(
            math.pi / 2.0,
            sinp,
        )
    else:
        pitch = math.asin(
            sinp
        )

    siny_cosp = 2.0 * (
        w * z + x * y
    )
    cosy_cosp = 1.0 - 2.0 * (
        y * y + z * z
    )
    yaw = math.atan2(
        siny_cosp,
        cosy_cosp,
    )

    return (
        float(roll),
        float(pitch),
        float(yaw),
    )


def quaternion_normalize(q):
    q = np.asarray(
        q,
        dtype=np.float64,
    )

    n = float(
        np.linalg.norm(q)
    )

    if n < 1e-12:
        return np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    return q / n


def quaternion_multiply(q1, q2):
    """
    Hamilton product for Isaac/USD [w, x, y, z] quaternions.
    """
    w1, x1, y1, z1 = quaternion_normalize(q1)
    w2, x2, y2, z2 = quaternion_normalize(q2)

    return quaternion_normalize(
        np.array(
            [
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ],
            dtype=np.float64,
        )
    )


def quaternion_conjugate(q):
    """
    Inverse rotation of a unit [w, x, y, z] quaternion.
    """
    w, x, y, z = quaternion_normalize(q)

    return np.array(
        [w, -x, -y, -z],
        dtype=np.float64,
    )


def quaternion_angle(q):
    """
    Rotation magnitude of a [w, x, y, z] quaternion, in radians.
    """
    w = float(
        abs(
            quaternion_normalize(q)[0]
        )
    )

    return 2.0 * math.acos(
        min(1.0, max(-1.0, w))
    )


def look_quaternion_from_points(
    eye,
    target,
    world_up=(0.0, 0.0, 1.0),
):
    """
    Orientation that points a camera at `target` from `eye`.

    Same convention as look_rotation_from_points, which builds the
    USD camera mount: local -Z looks forward, +Y is up. This one
    stays in numpy [w, x, y, z] so it can be composed with measured
    world poses.
    """
    eye = np.asarray(
        eye,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    forward = target - eye
    length = float(
        np.linalg.norm(forward)
    )

    if length < 1e-8:
        return np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    forward = forward / length

    # Camera looks along -Z, so its local Z axis points backwards.
    z_axis = -forward

    up = np.asarray(
        world_up,
        dtype=np.float64,
    )

    x_axis = np.cross(
        up,
        z_axis,
    )

    if float(np.linalg.norm(x_axis)) < 1e-8:
        # Looking straight up or down: pick another reference up.
        x_axis = np.cross(
            np.array([0.0, 1.0, 0.0]),
            z_axis,
        )

    x_axis = x_axis / float(
        np.linalg.norm(x_axis)
    )

    y_axis = np.cross(
        z_axis,
        x_axis,
    )

    y_axis = y_axis / float(
        np.linalg.norm(y_axis)
    )

    return rotation_matrix_to_quaternion(
        np.column_stack(
            [
                x_axis,
                y_axis,
                z_axis,
            ]
        )
    )


def rotation_matrix_to_quaternion(matrix):
    """
    3x3 rotation matrix to an Isaac/USD [w, x, y, z] quaternion.
    """
    m = np.asarray(
        matrix,
        dtype=np.float64,
    )

    trace = float(
        m[0, 0] + m[1, 1] + m[2, 2]
    )

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (
        m[0, 0] > m[1, 1]
        and m[0, 0] > m[2, 2]
    ):
        s = math.sqrt(
            1.0 + m[0, 0] - m[1, 1] - m[2, 2]
        ) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(
            1.0 + m[1, 1] - m[0, 0] - m[2, 2]
        ) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(
            1.0 + m[2, 2] - m[0, 0] - m[1, 1]
        ) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return quaternion_normalize(
        np.array(
            [w, x, y, z],
            dtype=np.float64,
        )
    )


def quaternion_from_two_vectors(source, target):
    """
    Minimal world-frame rotation taking vector source onto target.
    Returns [w, x, y, z].
    """
    a = np.asarray(
        source,
        dtype=np.float64,
    )
    b = np.asarray(
        target,
        dtype=np.float64,
    )

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na < 1e-9 or nb < 1e-9:
        raise ValueError(
            "Cannot align zero-length marker vector."
        )

    a = a / na
    b = b / nb

    dot = float(
        np.clip(
            np.dot(a, b),
            -1.0,
            1.0,
        )
    )

    if dot > 0.999999:
        return np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    if dot < -0.999999:
        # Choose a stable axis perpendicular to a.
        basis = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        )

        if abs(a[0]) > 0.9:
            basis = np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float64,
            )

        axis = np.cross(
            a,
            basis,
        )

        axis /= np.linalg.norm(
            axis
        )

        return np.array(
            [
                0.0,
                axis[0],
                axis[1],
                axis[2],
            ],
            dtype=np.float64,
        )

    cross = np.cross(
        a,
        b,
    )

    q = np.array(
        [
            1.0 + dot,
            cross[0],
            cross[1],
            cross[2],
        ],
        dtype=np.float64,
    )

    return quaternion_normalize(
        q
    )



def quaternion_to_rotation_matrix(
    q,
):
    """
    Convert Isaac quaternion [w, x, y, z] to a 3x3 rotation matrix.
    """
    q = quaternion_normalize(
        np.asarray(
            q,
            dtype=np.float64,
        )
    )

    w, x, y, z = q

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


def look_rotation_from_points(
    eye,
    target,
):
    """
    USD Camera looks along local -Z. Build a quat that points -Z
    from eye toward target, with world +Z as the preferred up.
    """
    eye = Gf.Vec3d(
        float(eye[0]),
        float(eye[1]),
        float(eye[2]),
    )

    target = Gf.Vec3d(
        float(target[0]),
        float(target[1]),
        float(target[2]),
    )

    forward = target - eye
    length = forward.GetLength()

    if length < 1e-8:
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0)

    forward = forward / length

    # Camera forward is -Z, so camera Z axis = -forward.
    z_axis = -forward

    world_up = Gf.Vec3d(0.0, 0.0, 1.0)

    x_axis = (
        world_up ^ z_axis
    )

    if x_axis.GetLength() < 1e-8:
        world_up = Gf.Vec3d(0.0, 1.0, 0.0)
        x_axis = world_up ^ z_axis

    x_axis = x_axis.GetNormalized()
    y_axis = (
        z_axis ^ x_axis
    ).GetNormalized()

    matrix = Gf.Matrix3d(
        x_axis[0], y_axis[0], z_axis[0],
        x_axis[1], y_axis[1], z_axis[1],
        x_axis[2], y_axis[2], z_axis[2],
    )

    quat = matrix.ExtractRotation().GetQuat()
    imag = quat.GetImaginary()

    return Gf.Quatf(
        float(quat.GetReal()),
        float(imag[0]),
        float(imag[1]),
        float(imag[2]),
    )
