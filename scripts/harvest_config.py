ARM_PATH = "/World/HarvestBot/jackal/ur5e/base_link"
JACKAL_PATH = "/World/HarvestBot/jackal"

USD_GRASP_TIP_PATH = (
    "/World/HarvestBot/jackal/ur5e/Gripper/"
    "Robotiq_Hand_E_edit/base_link/GraspTip"
)

URDF_PATH = "/home/charlotte/harvestloop_sim/assets/ur5e_scaled_06.urdf"
ROBOT_DESCRIPTION_PATH = "/home/charlotte/harvestloop_sim/assets/ur5e_robot_description_scaled_06.yaml"
RMPFLOW_CONFIG_PATH = "/home/charlotte/harvestloop_sim/assets/ur5e_rmpflow_config_scaled_06.yaml"

PHYSICS_DT = 1.0 / 60.0

DRIVE_STIFFNESS = 10000.0
DRIVE_DAMPING = 500.0
DRIVE_MAX_FORCE = 250.0

PREGRASP_OFFSET = 0.06
PREGRASP_TOLERANCE = 0.025
MAX_TARGET_DISTANCE = 0.55
