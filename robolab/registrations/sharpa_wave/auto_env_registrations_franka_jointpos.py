# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment registration for the Franka + SharpaWave composite robot.

Mirrors the DROID registration layout (over-shoulder-left ego camera + hand
wrist camera) since the physical rig is a DROID-style Franka cell with the
Robotiq swapped for a SharpaWave hand.
"""

import robolab.constants
from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR


def auto_register_franka_sharpa_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=None,
                                     finger_actions=False):
    """Register tasks for the Franka+SharpaWave robot.

    Args:
        task_dirs: Subdirectories to search for tasks.
        task: Optional task name(s) to restrict registration to.
        finger_actions: If True use the 29-dim per-finger action space,
            else the 8-dim arm + binary-hand space.
    """
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras,
        generate_obs_cfg,
    )
    from robolab.robots.franka_sharpa_wave import (
        FrankaSharpaFingerActionCfg,
        FrankaSharpaJointPositionActionCfg,
        FrankaSharpaProprioceptionObservationCfg,
        FrankaSharpaWaveCfg,
        FrankaSharpaWristCameraCfg,
        contact_gripper,
    )
    from robolab.registrations.sharpa_wave.auto_env_registrations_gr00t import RobotHeadCameraCfg
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    # Ego camera above the robot base looking forward at the table — matches
    # the physical rig's camera placement.
    cameras = [RobotHeadCameraCfg, FrankaSharpaWristCameraCfg]

    ImageObsCfg = generate_image_obs_from_cameras(cameras)
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])

    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": FrankaSharpaProprioceptionObservationCfg(),
        "viewport_cam": ViewportCameraCfg(),
    })

    actions_cfg = FrankaSharpaFingerActionCfg() if finger_actions else FrankaSharpaJointPositionActionCfg()

    # Wrist cameras are robot-mounted — exclude from scene camera mixins.
    scene_cameras = [RobotHeadCameraCfg]

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=task_dirs,
        tasks=task,
        pattern="*.py",
        env_prefix="",
        env_postfix="",
        observations_cfg=ObservationCfg(),
        actions_cfg=actions_cfg,
        robot_cfg=FrankaSharpaWaveCfg,
        camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
    )

    if robolab.constants.VERBOSE:
        from robolab.core.environments.factory import print_env_table
        print_env_table()
