# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FR3 Duo torso with two Panda arms and left+right SharpaWave hands.

Mirrors the physical lab rig: an FR3 Duo (two arms on one torso mount, official
mount kinematics from frankarobotics/franka_description
accessories/fr3_duo_mount_v0_3) with a SharpaWave hand on each flange. The
composite USD (assets/robots/franka_duo_sharpa_wave.usda) is a single
articulation: torso + 2×7 arm DoF + 2×22 finger DoF = 58 actuated joints.
Panda arms stand in for FR3s (near-identical kinematics).
"""

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from robolab.constants import ROBOTS_DIR
from robolab.robots.droid import BinaryJointPositionZeroToOneActionCfg, _to_torch
from robolab.robots.sharpa_wave import HAND_JOINTS_ORDERED as RIGHT_HAND_JOINTS_ORDERED

LEFT_HAND_JOINTS_ORDERED = [n.replace("right_", "left_") for n in RIGHT_HAND_JOINTS_ORDERED]
LEFT_ARM_JOINTS = [f"left_panda_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"right_panda_joint{i}" for i in range(1, 8)]

_FINGERS_EXPR = ["(left|right)_(thumb|index|middle|ring|pinky)_.*"]


def _open_close(side: str):
    open_cmd = {f"{side}_(thumb|index|middle|ring|pinky)_.*": 0.0}
    close_cmd = {
        f"{side}_thumb_CMC_FE": 1.2,
        f"{side}_thumb_CMC_AA": 0.25,
        f"{side}_thumb_MCP_FE": 1.0,
        f"{side}_thumb_MCP_AA": 0.0,
        f"{side}_thumb_IP": 1.2,
        f"{side}_(index|middle|ring|pinky)_MCP_FE": 1.3,
        f"{side}_(index|middle|ring|pinky)_MCP_AA": 0.0,
        f"{side}_(index|middle|ring|pinky)_PIP": 1.4,
        f"{side}_(index|middle|ring|pinky)_DIP": 1.1,
        f"{side}_pinky_CMC": 0.2,
    }
    return open_cmd, close_cmd


# Wrist cameras from chy-applovin/flow-policy#4 RIGID_WRIST_CAM — grasp-aperture
# mounts, fovy 90. Deliberately NOT mirrors of each other (asymmetric hand roles).
_RIGHT_WRIST_CAM = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/right_hand/right_hand_C_MC/wrist_cam_right",
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.8,
        focus_distance=28.0,
        horizontal_aperture=7.4667,
        vertical_aperture=5.6,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0747, -0.0213, -0.0355),
        rot=(0.2051, 0.5957, -0.7446, 0.2205),
        convention="opengl",
    ),
)

_LEFT_WRIST_CAM = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/left_hand/left_hand_C_MC/wrist_cam_left",
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.8,
        focus_distance=28.0,
        horizontal_aperture=7.4667,
        vertical_aperture=5.6,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        # Exact y-mirror of the right camera (hardware-symmetric mounts, the
        # standard on bimanual rigs — ALOHA 2, FR3 Duo). The Sharpa left hand
        # is a y-mirror of the right in its own base frame, so this aims at
        # the left thumb–index grasp aperture. Replaces flow-policy#4's
        # asymmetric left pose, which was tuned for TACO's task-specific
        # left-palm-inward role rather than mirrored hardware.
        pos=(0.0747, 0.0213, -0.0355),
        rot=(-0.2051, 0.5957, 0.7446, 0.2205),
        convention="opengl",
    ),
)


@configclass
class FrankaDuoSharpaWaveCfg:
    """FR3 Duo + dual SharpaWave composite articulation."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ROBOTS_DIR, "franka_duo_sharpa_wave.usda"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Torso mount height/offset — the FR3 Duo pedestal places the mount
            # above the table plane; tuned via examples/dump_duo_poses.py
            # sweeps (first guess −0.05/0.55 put the shoulders over the table).
            pos=(-0.35, 0, 0.45),
            rot=(1, 0, 0, 0),
            joint_pos={
                # "Fold" ready pose (candidate g from examples/dump_duo_poses.py
                # sweep) with a mirrored joint-1 split: with j1=0 the yawed-in
                # shoulder mounts converge both wrists to the midline (hands
                # overlapped). ±0.5 on joint 1 keeps each wrist over its own
                # half of the table.
                "left_panda_joint1": 0.5,
                "right_panda_joint1": -0.5,
                "(left|right)_panda_joint2": 0.3,
                "(left|right)_panda_joint3": 0.0,
                "(left|right)_panda_joint4": -2.2,
                "(left|right)_panda_joint5": 0.0,
                "(left|right)_panda_joint6": 2.0,
                "(left|right)_panda_joint7": 0.0,
                "(left|right)_(thumb|index|middle|ring|pinky)_.*": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1,
        actuators={
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "forearms": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,
                damping=80.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=_FINGERS_EXPR,
                stiffness=None,
                damping=None,
            ),
        },
    )

    right_wrist_cam = _RIGHT_WRIST_CAM
    left_wrist_cam = _LEFT_WRIST_CAM


@configclass
class FrankaDuoWristCamerasCfg:
    """Introspection wrapper for generate_image_obs_from_cameras."""

    right_wrist_cam = _RIGHT_WRIST_CAM
    left_wrist_cam = _LEFT_WRIST_CAM


########################################################
# Contact gripper
########################################################

# ContactSensor allows exactly one prim per env; use the right index fingertip
# (the right hand leads most single-object grasps).
contact_gripper = {"gripper": "{ENV_REGEX_NS}/robot/right_hand/right_index_elastomer"}


########################################################
# Actions
########################################################

_R_OPEN, _R_CLOSE = _open_close("right")
_L_OPEN, _L_CLOSE = _open_close("left")


@configclass
class FrankaDuoSharpaJointPositionActionCfg:
    """[7 left arm, 7 right arm, left binary hand, right binary hand] = 16-dim."""

    left_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEFT_ARM_JOINTS,
        preserve_order=True,
        use_default_offset=False,
    )

    right_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINTS,
        preserve_order=True,
        use_default_offset=False,
    )

    left_hand = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["left_(thumb|index|middle|ring|pinky)_.*"],
        open_command_expr=_L_OPEN,
        close_command_expr=_L_CLOSE,
    )

    right_hand = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["right_(thumb|index|middle|ring|pinky)_.*"],
        open_command_expr=_R_OPEN,
        close_command_expr=_R_CLOSE,
    )


@configclass
class FrankaDuoSharpaFingerActionCfg:
    """[7 left arm, 7 right arm, 22 left fingers, 22 right fingers] = 58-dim."""

    left_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEFT_ARM_JOINTS,
        preserve_order=True,
        use_default_offset=False,
    )

    right_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINTS,
        preserve_order=True,
        use_default_offset=False,
    )

    left_hand = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEFT_HAND_JOINTS_ORDERED,
        preserve_order=True,
        use_default_offset=False,
    )

    right_hand = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_HAND_JOINTS_ORDERED,
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class FrankaDuoSharpaIKActionCfg:
    """Absolute wrist-pose IK per arm + per-finger hands (7+7+22+22 = 58-dim).

    Pose commands are (x, y, z, qw, qx, qy, qz) for each flange (panda_link8)
    in the ROBOT ROOT frame (the torso at init_state.pos). Used by the GR00T
    sharpa client, whose model outputs wrist poses rather than arm joints.
    """

    left_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=LEFT_ARM_JOINTS,
        body_name="left_panda_link8",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )

    right_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINTS,
        body_name="right_panda_link8",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )

    left_hand = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEFT_HAND_JOINTS_ORDERED,
        preserve_order=True,
        use_default_offset=False,
    )

    right_hand = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_HAND_JOINTS_ORDERED,
        preserve_order=True,
        use_default_offset=False,
    )


########################################################
# Observations
########################################################


def _joint_pos(env, names):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(names, preserve_order=True)
    return _to_torch(robot.data.joint_pos)[:, ids]


def left_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _joint_pos(env, LEFT_ARM_JOINTS)


def right_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _joint_pos(env, RIGHT_ARM_JOINTS)


def left_hand_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _joint_pos(env, LEFT_HAND_JOINTS_ORDERED)


def right_hand_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _joint_pos(env, RIGHT_HAND_JOINTS_ORDERED)


def _body_pose(env, body_name):
    robot = env.scene["robot"]
    idx = robot.data.body_names.index(body_name)
    pos = _to_torch(robot.data.body_pos_w)[:, idx, :] - env.scene.env_origins[:, 0:3]
    quat = _to_torch(robot.data.body_quat_w)[:, idx, :]
    return pos, quat


def left_eef_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _body_pose(env, "left_panda_link8")[0]


def left_eef_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _body_pose(env, "left_panda_link8")[1]


def right_eef_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _body_pose(env, "right_panda_link8")[0]


def right_eef_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return _body_pose(env, "right_panda_link8")[1]


@configclass
class FrankaDuoProprioceptionObservationCfg(ObsGroup):
    left_arm_joint_pos = ObsTerm(func=left_arm_joint_pos)
    right_arm_joint_pos = ObsTerm(func=right_arm_joint_pos)
    left_hand_joint_pos = ObsTerm(func=left_hand_joint_pos)
    right_hand_joint_pos = ObsTerm(func=right_hand_joint_pos)
    # Flange (panda_link8) poses in the env-local frame, w-first quats.
    left_eef_pos = ObsTerm(func=left_eef_pos)
    left_eef_quat = ObsTerm(func=left_eef_quat)
    right_eef_pos = ObsTerm(func=right_eef_pos)
    right_eef_quat = ObsTerm(func=right_eef_quat)

    def __post_init__(self) -> None:
        self.enable_corruption = False  # must include
        self.concatenate_terms = False  # must include
