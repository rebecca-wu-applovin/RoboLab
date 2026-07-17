# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""
Hand-toggle episode for the FR3 Duo + dual SharpaWave bimanual robot.

Holds both arms at their current joint positions while toggling the two hands
in ALTERNATION (left fist while right open, and vice versa) every
`--toggle-every` steps — verifies the two hands are independently controlled.

Usage:
    $ python examples/run_franka_duo_toggle.py --headless

Output:
    Per-env videos saved to output/run_franka_duo_toggle/<task_env>/
"""

import argparse
import cv2  # noqa: F401  must be imported before isaaclab
import os
import re
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run FR3 Duo + SharpaWave hand-toggle episode.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs="+", default=None)
parser.add_argument("--num-steps", type=int, default=100)
parser.add_argument("--toggle-every", type=int, default=15)
parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                    choices=["all", "viewport", "sensor", "none"])

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = args_cli.video_mode != "none"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from robolab.constants import PACKAGE_DIR, get_output_dir, set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.registrations.sharpa_wave.auto_env_registrations_duo import (  # noqa: E402
    auto_register_franka_duo_envs,
)
from robolab.robots.franka_duo_sharpa_wave import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS  # noqa: E402


def run_toggle_episode(env, env_cfg=None, *, save_videos=True, video_mode="all",
                       num_steps=100, toggle_every=15):
    robot = env.scene["robot"]
    obs, _ = env.reset()

    print(f"Articulation joints ({len(robot.data.joint_names)}): {robot.data.joint_names}")
    arm_ids, _ = robot.find_joints(LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS, preserve_order=True)
    arm_ids = torch.as_tensor(arm_ids, device=env.device)

    instruction = getattr(env_cfg, "instruction", None) or "duo_toggle"
    if isinstance(instruction, dict):
        instruction = instruction.get("default", "duo_toggle")
    cleaned = re.sub(r"[^\w\s]", "", instruction).replace(" ", "_")

    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt) if env_cfg is not None else 15
    save_sensor = save_videos and video_mode in ("all", "sensor")
    save_viewport = save_videos and video_mode in ("all", "viewport")

    writers_obs, writers_vp = [], []
    if save_videos:
        for env_id in range(env.num_envs):
            suffix = f"_env{env_id}" if env.num_envs > 1 else ""
            if save_sensor:
                writers_obs.append(VideoWriter(os.path.join(get_output_dir(), f"{cleaned}{suffix}.mp4"), video_fps))
            if save_viewport:
                writers_vp.append(VideoWriter(os.path.join(get_output_dir(), f"{cleaned}{suffix}_viewport.mp4"), video_fps))

    phase = False
    try:
        for count in tqdm(range(num_steps)):
            if count % toggle_every == 0:
                phase = not phase
                print(f"[Step {count:04d}] left={'fist' if phase else 'open'} right={'open' if phase else 'fist'}")

            arm_pose = robot.data.joint_pos[:, arm_ids]
            left = torch.full((env.num_envs, 1), 1.0 if phase else 0.0, device=env.device)
            right = 1.0 - left
            actions = torch.cat([arm_pose, left, right], dim=1)

            obs, _, term, trunc, info = env.step(actions)

            if save_videos:
                for env_id in range(env.num_envs):
                    if save_sensor:
                        frame = unpack_image_obs(obs, scale=0.5, env_id=env_id).get("combined_image")
                        if frame is not None:
                            writers_obs[env_id].write(frame)
                    if save_viewport:
                        frame_vp = unpack_viewport_cams(obs, env_id=env_id).get("combined_image")
                        if frame_vp is not None:
                            writers_vp[env_id].write(frame_vp)
    finally:
        for vw in writers_obs + writers_vp:
            try:
                vw.release()
            except Exception:
                pass

    return True


def main():
    output_dir = os.path.join(PACKAGE_DIR, "output", "run_franka_duo_toggle")
    os.makedirs(output_dir, exist_ok=True)

    tasks = args_cli.task or ["BananaInBowlTask"]
    auto_register_franka_duo_envs(task=tasks)
    task_envs = get_envs(task=tasks)
    print(f"Running FR3 Duo hand toggle on {len(task_envs)} environments: {task_envs}")

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        env, env_cfg = create_env(task_env, device=args_cli.device,
                                  num_envs=args_cli.num_envs, use_fabric=True)
        try:
            print(f"Running {task_env}: '{env_cfg.instruction}'")
            run_toggle_episode(env, env_cfg,
                               save_videos=args_cli.save_videos,
                               video_mode=args_cli.video_mode,
                               num_steps=args_cli.num_steps,
                               toggle_every=args_cli.toggle_every)
            end_episode(env)
        finally:
            env.close()

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Terminated with error: {e}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
