# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Franka Panda arm with a SharpaWave right hand bolted to the flange.

Mirrors the physical lab rig: the Robotiq gripper of the DROID setup is
replaced by Sharpa's ``right_sharpa_wave_with_flange`` asset, welded to
panda_link8 with the same −45° flange rotation the Robotiq used. The composite
USD (assets/robots/franka_sharpa_wave.usda) is a single articulation:
7 arm DoF + 22 finger DoF.
"""

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
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
from robolab.robots.sharpa_wave import (
    _CLOSE_CMD,
    _OPEN_CMD,
    FINGER_JOINTS_EXPR,
    HAND_JOINTS_ORDERED,
)

# Wrist cameras on the hand base link (frame: fingers +z, thumb +y, palm +x).
# Poses are calibration starting points — iterate with
# examples/dump_sharpa_cameras.py --franka and adjust to match the real rig.
# Pose from chy-applovin/flow-policy#4 (RIGID_WRIST_CAM["right"], user-validated
# 2026-07-10 on the same right_hand_C_MC body): radial-dorsal mount aimed at the
# thumb–index grasp aperture, advanced 5 cm along the view ray, fovy 90.
# MuJoCo cameras share the opengl convention (−Z forward, +Y up), so pos/quat
# transfer directly.
_WRIST_CAM = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/sharpa/right_hand_C_MC/wrist_cam",
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

_WRIST_CAM_2 = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/sharpa/right_hand_C_MC/wrist_cam_2",
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
        # thumb side of the wrist, looking across the palm workspace
        pos=(0.03, 0.14, -0.02),
        rot=(-0.1696, 0.6865, -0.6652, -0.2397),
        convention="opengl",
    ),
)


@configclass
class FrankaSharpaWaveCfg:
    """Franka + SharpaWave composite articulation."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ROBOTS_DIR, "franka_sharpa_wave.usda"),
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
                # Same ready pose as the DROID config.
                "panda_joint1": 0.0,
                "panda_joint2": -1 / 5 * np.pi,
                "panda_joint3": 0.0,
                "panda_joint4": -4 / 5 * np.pi,
                "panda_joint5": 0.0,
                "panda_joint6": 3 / 5 * np.pi,
                "panda_joint7": 0.0,
                "right_(thumb|index|middle|ring|pinky)_.*": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1,
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,
                damping=80.0,
            ),
            # Sharpa ships position-mode gains in the USD; None inherits them.
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=FINGER_JOINTS_EXPR,
                stiffness=None,
                damping=None,
            ),
        },
    )

    wrist_cam = _WRIST_CAM
    wrist_cam_2 = _WRIST_CAM_2


@configclass
class FrankaSharpaWristCameraCfg:
    """Introspection wrapper for generate_image_obs_from_cameras."""

    wrist_cam = _WRIST_CAM
    wrist_cam_2 = _WRIST_CAM_2


########################################################
# Contact gripper
########################################################

contact_gripper = {"gripper": "{ENV_REGEX_NS}/robot/sharpa/right_index_elastomer"}


########################################################
# Actions
########################################################


@configclass
class FrankaSharpaJointPositionActionCfg:
    """7 arm joint targets + binary hand (open flat / power fist).

    8-dim, mirroring the built-in DROID convention so gripper-toggle sanity
    checks and binary-gripper policy plumbing work unchanged.
    """

    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
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
class FrankaSharpaFingerActionCfg:
    """7 arm joint targets + 22 per-finger targets (29-dim), for dexterous policies."""

    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
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


def arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    ids, _ = robot.find_joints([f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)
    return _to_torch(robot.data.joint_pos)[:, ids]


def hand_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    ids, _ = robot.find_joints(HAND_JOINTS_ORDERED, preserve_order=True)
    return _to_torch(robot.data.joint_pos)[:, ids]


@configclass
class FrankaSharpaProprioceptionObservationCfg(ObsGroup):
    arm_joint_pos = ObsTerm(func=arm_joint_pos)
    hand_joint_pos = ObsTerm(func=hand_joint_pos)

    def __post_init__(self) -> None:
        self.enable_corruption = False  # must include
        self.concatenate_terms = False  # must include
