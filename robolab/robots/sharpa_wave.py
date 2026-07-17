# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SharpaWave dexterous hand (right, float base) robot configuration.

Uses the official Sharpa Robotics USD asset (assets/robots/sharpa_wave/,
vendored from https://github.com/sharpa-robotics/sharpa-urdf-usd-xml). The
float-base variant adds 6 virtual DoF (x/y/z prismatic + roll/pitch/yaw
continuous) in front of the wrist so the hand pose can be commanded directly
without an arm. Finger dynamics are pre-calibrated in the USD for IsaacLab
position mode, so the finger actuators inherit stiffness/damping from USD.
"""

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from robolab.constants import ROBOTS_DIR
from robolab.robots.droid import BinaryJointPositionZeroToOneActionCfg, _to_torch

# Order defines the layout of the 6-dim body action (preserve_order=True below).
FLOAT_BASE_JOINTS = [
    "right_x_joint",
    "right_y_joint",
    "right_z_joint",
    "right_roll_joint",
    "right_pitch_joint",
    "right_yaw_joint",
]

# All 22 actuated finger joints (fixed elastomer/fingertip joints are not articulation joints).
FINGER_JOINTS_EXPR = ["right_(thumb|index|middle|ring|pinky)_.*"]

# Canonical per-finger joint order, thumb→pinky in URDF document order. Used for
# the GR00T N1.7 `real_r1_pro_sharpa_relative_eef` embodiment's 22-dim
# right_hand_joints state/action vector. ASSUMPTION: GR00T's pretraining data
# follows the vendor URDF order — unverified against the R1 Pro teleop stack;
# if fingers move "shuffled" under the policy, this list is the first suspect.
HAND_JOINTS_ORDERED = [
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
]

# Binary hand commands: open = flat hand, close = power-grasp fist.
# Close targets sit inside the URDF limits (flexion joints ~80% of upper).
_OPEN_CMD = {"right_(thumb|index|middle|ring|pinky)_.*": 0.0}
_CLOSE_CMD = {
    "right_thumb_CMC_FE": 1.2,
    "right_thumb_CMC_AA": 0.25,
    "right_thumb_MCP_FE": 1.0,
    "right_thumb_MCP_AA": 0.0,
    "right_thumb_IP": 1.2,
    "right_(index|middle|ring|pinky)_MCP_FE": 1.3,
    "right_(index|middle|ring|pinky)_MCP_AA": 0.0,
    "right_(index|middle|ring|pinky)_PIP": 1.4,
    "right_(index|middle|ring|pinky)_DIP": 1.1,
    "right_pinky_CMC": 0.2,
}


# Wrist camera for the GR00T sharpa embodiment ("right_wrist_view"). Pose from
# chy-applovin/flow-policy#4 RIGID_WRIST_CAM["right"] (user-validated on the
# same right_hand_C_MC body): aimed at the thumb–index grasp aperture, fovy 90.
_SHARPA_WRIST_CAM = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/right_hand_C_MC/wrist_cam",
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


@configclass
class SharpaWaveCfg:
    """Cfg class that adds the SharpaWave right hand (float base) to scene configurations."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ROBOTS_DIR, "sharpa_wave", "right_sharpa_wave_with_float_base.usda"),
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
            pos=(0, 0, 0),
            rot=(1, 0, 0, 0),
            joint_pos={
                # GR00T-N1.7 sharpa-embodiment training-mean wrist pose, mapped
                # into the sim frame via SIM_TO_R1PRO_POS (see
                # policies/gr00t/sharpa_client.py). Keeps the initial state
                # in-distribution for the policy; also a sensible hover for the
                # toggle demo (~19 cm above table center, fingers toward table).
                "right_x_joint": 0.549,
                "right_y_joint": -0.124,
                "right_z_joint": 0.188,
                "right_roll_joint": -1.6651,
                "right_pitch_joint": 1.3607,
                "right_yaw_joint": 1.0561,
                "right_(thumb|index|middle|ring|pinky)_.*": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1,
        actuators={
            # Virtual wrist joints: no USD calibration applies, use stiff position control.
            "float_base": ImplicitActuatorCfg(
                joint_names_expr=["right_(x|y|z|roll|pitch|yaw)_joint"],
                effort_limit=200.0,
                velocity_limit=2.0,
                stiffness=1000.0,
                damping=100.0,
            ),
            # Sharpa ships position-mode gains in the USD; None inherits them.
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=FINGER_JOINTS_EXPR,
                stiffness=None,
                damping=None,
            ),
        },
    )

    wrist_cam = _SHARPA_WRIST_CAM


@configclass
class SharpaWristCameraCfg:
    """Introspection wrapper so the wrist camera can be passed to
    generate_image_obs_from_cameras (same pattern as droid.WristCameraCfg).
    The scene's wrist_cam is still sourced from SharpaWaveCfg."""

    wrist_cam = _SHARPA_WRIST_CAM


########################################################
# Contact gripper
########################################################

# IsaacLab ContactSensor requires exactly one prim per env for
# filter_prim_paths_expr to work (see robolab/robots/droid.py). Use the index
# fingertip elastomer — the tactile pad that leads most grasps.
contact_gripper = {"gripper": "{ENV_REGEX_NS}/robot/right_index_elastomer"}


########################################################
# Actions
########################################################


@configclass
class SharpaWaveJointPositionActionCfg:
    """6-DoF wrist pose (joint targets on the float base) + binary hand open/close.

    Action layout: [x, y, z, roll, pitch, yaw, hand] with hand in [0, 1]
    (>0.5 closes to a fist, else opens flat) — mirrors the built-in binary
    gripper convention so existing eval plumbing works unchanged. Per-finger
    control needs a dedicated action cfg replacing the binary hand term.
    """

    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=FLOAT_BASE_JOINTS,
        preserve_order=True,
        use_default_offset=False,
    )

    hand = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=FINGER_JOINTS_EXPR,
        open_command_expr=_OPEN_CMD,
        close_command_expr=_CLOSE_CMD,
    )


@configclass
class SharpaWaveGR00TActionCfg:
    """Per-finger control for the GR00T sharpa embodiment.

    Action layout: [x, y, z, roll, pitch, yaw, 22 hand joints] (28-dim), wrist
    as absolute float-base joint targets, fingers as absolute joint positions in
    HAND_JOINTS_ORDERED order.
    """

    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=FLOAT_BASE_JOINTS,
        preserve_order=True,
        use_default_offset=False,
    )

    hand = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=HAND_JOINTS_ORDERED,
        preserve_order=True,
        use_default_offset=False,
    )


########################################################
# Observations
########################################################


def wrist_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Float-base joint positions [x, y, z, roll, pitch, yaw] — the wrist pose."""
    robot = env.scene[asset_cfg.name]
    ids, _ = robot.find_joints(FLOAT_BASE_JOINTS, preserve_order=True)
    return _to_torch(robot.data.joint_pos)[:, ids]


def hand_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """22 finger joint positions in HAND_JOINTS_ORDERED order."""
    robot = env.scene[asset_cfg.name]
    ids, _ = robot.find_joints(HAND_JOINTS_ORDERED, preserve_order=True)
    return _to_torch(robot.data.joint_pos)[:, ids]


@configclass
class SharpaProprioceptionObservationCfg(ObsGroup):
    wrist_joint_pos = ObsTerm(func=wrist_joint_pos)
    hand_joint_pos = ObsTerm(func=hand_joint_pos)

    def __post_init__(self) -> None:
        self.enable_corruption = False  # must include
        self.concatenate_terms = False  # must include
