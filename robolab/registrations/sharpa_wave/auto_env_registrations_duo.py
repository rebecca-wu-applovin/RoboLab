# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment registration for the FR3 Duo + dual SharpaWave bimanual robot.

Camera layout matches the GR00T sharpa embodiment / physical rig: one ego
camera above the torso plus left and right grasp-aperture wrist cameras.
"""

import robolab.constants
from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR


def auto_register_franka_duo_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=None,
                                  finger_actions=False):
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras,
        generate_obs_cfg,
    )
    from robolab.registrations.sharpa_wave.auto_env_registrations_gr00t import RobotHeadCameraCfg
    from robolab.robots.franka_duo_sharpa_wave import (
        FrankaDuoProprioceptionObservationCfg,
        FrankaDuoSharpaFingerActionCfg,
        FrankaDuoSharpaJointPositionActionCfg,
        FrankaDuoSharpaWaveCfg,
        FrankaDuoWristCamerasCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    cameras = [RobotHeadCameraCfg, FrankaDuoWristCamerasCfg]

    ImageObsCfg = generate_image_obs_from_cameras(cameras)
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])

    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": FrankaDuoProprioceptionObservationCfg(),
        "viewport_cam": ViewportCameraCfg(),
    })

    actions_cfg = FrankaDuoSharpaFingerActionCfg() if finger_actions else FrankaDuoSharpaJointPositionActionCfg()

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
        robot_cfg=FrankaDuoSharpaWaveCfg,
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
