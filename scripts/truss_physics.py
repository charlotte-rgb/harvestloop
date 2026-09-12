from pxr import UsdPhysics, Sdf


def add_collider(prim):
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)


def make_truss_attached(truss_prim):
    """
    Truss is a rigid body but kinematic:
    it stays exactly where scene generation placed it.
    """
    rigid = UsdPhysics.RigidBodyAPI.Apply(truss_prim)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(True)
    return rigid


def detach_truss(truss_prim):
    """
    Simulated cut:
    stop holding the truss kinematically.
    Gravity/physics now affects it.
    """
    rigid = UsdPhysics.RigidBodyAPI(truss_prim)

    if not rigid:
        raise RuntimeError(
            f"{truss_prim.GetPath()} is not a rigid body"
        )

    rigid.GetKinematicEnabledAttr().Set(False)


def create_fixed_joint(
    stage,
    joint_path,
    body0_path,
    body1_path,
):
    """
    Used later for:
        gripper <-> truss
    """
    if stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)

    joint = UsdPhysics.FixedJoint.Define(
        stage,
        joint_path,
    )

    joint.CreateBody0Rel().SetTargets([
        Sdf.Path(body0_path)
    ])

    joint.CreateBody1Rel().SetTargets([
        Sdf.Path(body1_path)
    ])

    return joint


def remove_joint(stage, joint_path):
    if stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)
        return True

    return False
