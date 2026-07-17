# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment registration for SharpaWave driven by GR00T N1.7's sharpa embodiment.

Differences from auto_env_registrations_jointpos.py:
- per-finger 28-dim action space (SharpaWaveGR00TActionCfg) instead of the
  binary open/fist hand,
- the hand's wrist camera feeds image observations (GR00T's
  right_wrist_view) alongside one exterior camera (GR00T's ego_view),
- a proprioception group exposing wrist pose + 22 finger joints.
"""

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

import robolab.constants
from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR


@configclass
class RobotHeadCameraCfg:
    """Robot-side ego view approximating the R1 Pro head camera: above the
    robot base looking forward and down at the table (GR00T's ego_view)."""

    # Head-cam v3 from chy-applovin/flow-policy#4 (locked 2026-07-14 after
    # occlusion/framing sweeps: fwd 0.16, right 0.05, 0.53 m above mean wrist
    # height, pitch 64° down, fovy 60 — 0/937 frames with content clipped).
    # Their aligned frame sits +1.0 m above our table frame (wrists ~1.17 vs
    # our ~0.17), so height 1.7 maps to 0.70 here.
    robot_head_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot_head_camera",
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=4.3109,
            vertical_aperture=3.2332,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            # Candidate f from the 6-pose ego sweep (2026-07-16,
            # output/camera_calibration/ego_sweep_*.png): above the base,
            # offset to the robot's left, aimed at table center. Unlike the
            # flow-policy#4 head-cam (floating hands, no arm), the Franka arm
            # occludes the workspace from any on-axis mount — the lateral
            # offset keeps banana/bowl/table visible with the arm frame-right.
            pos=(-0.15, 0.35, 1.15),
            rot=(-0.5024, -0.1547, 0.2504, 0.813),
            convention="opengl",
        ),
    )


def auto_register_sharpa_wave_gr00t_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=None):
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras,
        generate_obs_cfg,
    )
    from robolab.robots.sharpa_wave import (
        SharpaProprioceptionObservationCfg,
        SharpaWaveCfg,
        SharpaWaveGR00TActionCfg,
        SharpaWristCameraCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    cameras = [RobotHeadCameraCfg, SharpaWristCameraCfg]

    ImageObsCfg = generate_image_obs_from_cameras(cameras)
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])

    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": SharpaProprioceptionObservationCfg(),
        "viewport_cam": ViewportCameraCfg(),
    })

    # SharpaWristCameraCfg is robot-mounted (wrist_cam is already attached via
    # SharpaWaveCfg) — keep it out of the scene camera mixins so it doesn't
    # spawn before its parent prim exists (same as droid registration).
    scene_cameras = [RobotHeadCameraCfg]

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=task_dirs,
        tasks=task,
        pattern="*.py",
        env_prefix="",
        env_postfix="",
        observations_cfg=ObservationCfg(),
        actions_cfg=SharpaWaveGR00TActionCfg(),
        robot_cfg=SharpaWaveCfg,
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
