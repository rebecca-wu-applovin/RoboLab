# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment registration for the SharpaWave dexterous hand (float base).

Mirrors robolab/registrations/droid/auto_env_registrations_jointpos.py, with
two departures: the SharpaWave asset has no wrist camera, so only exterior
cameras feed the image observations, and there is no proprioception group yet
(the droid one is Franka-specific).
"""

import robolab.constants
from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR


def auto_register_sharpa_wave_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=None, cameras=None):
    """Automatically discover and register tasks for the SharpaWave hand.

    Args:
        task_dirs: Subdirectories to search for tasks.
        task: If provided, only register the specified task(s). Accepts a single
              task name/filename/path (str) or a list of them.
        cameras: List of exterior camera config classes observed by the policy.
              Defaults to over-shoulder left + right.
    """
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras,
        generate_obs_cfg,
    )
    from robolab.robots.sharpa_wave import (
        SharpaWaveCfg,
        SharpaWaveJointPositionActionCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import (
        EgocentricMirroredCameraCfg,
        OverShoulderLeftCameraCfg,
        OverShoulderRightCameraCfg,
    )
    from robolab.variations.lighting import SphereLightCfg

    if cameras is None:
        cameras = [OverShoulderLeftCameraCfg, OverShoulderRightCameraCfg]

    ImageObsCfg = generate_image_obs_from_cameras(cameras)
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])

    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "viewport_cam": ViewportCameraCfg(),
    })

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=task_dirs,
        tasks=task,
        pattern="*.py",
        env_prefix="",
        env_postfix="",
        observations_cfg=ObservationCfg(),
        actions_cfg=SharpaWaveJointPositionActionCfg(),
        robot_cfg=SharpaWaveCfg,
        camera_cfg=[*cameras, EgocentricMirroredCameraCfg],
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
