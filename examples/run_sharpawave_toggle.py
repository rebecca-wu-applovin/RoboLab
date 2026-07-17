# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""
Run a hand-toggle episode with the SharpaWave dexterous hand (float base).

Holds the 6-DoF virtual wrist at its current pose while toggling all 22
finger joints between a flat hand and a power-grasp fist every
`--toggle-every` steps. The SharpaWave analog of run_gripper_toggle.py —
sanity-checks the hand action path on any registered task.

Usage:
    Basic usage (default task: BananaInBowlTask):
    $ python examples/run_sharpawave_toggle.py --headless

    Specific task:
    $ python examples/run_sharpawave_toggle.py --task RubiksCubeTask --headless

Output:
    Per-env videos saved to output/run_sharpawave_toggle/<task_env>/
"""

import argparse
import cv2  # noqa: F401  must be imported before isaaclab
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run SharpaWave hand-toggle episode on a registered task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs="+", default=None,
                    help="List of tasks to run on (default: BananaInBowlTask).")
parser.add_argument("--num-steps", type=int, default=100, help="Number of steps per episode.")
parser.add_argument("--toggle-every", type=int, default=15, help="Toggle hand every N steps.")
parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                    choices=["all", "viewport", "sensor", "none"],
                    help="Which videos to save (default: all)")

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
from robolab.registrations.sharpa_wave.auto_env_registrations_jointpos import auto_register_sharpa_wave_envs  # noqa: E402
from robolab.robots.sharpa_wave import FLOAT_BASE_JOINTS  # noqa: E402

import re  # noqa: E402


def run_hand_toggle_episode(env, env_cfg=None, *, save_videos=True, video_mode="all",
                            num_steps=100, toggle_every=15):
    """Toggle the hand open/closed while holding the float-base wrist pose."""
    robot = env.scene["robot"]
    obs, _ = env.reset()

    print(f"Articulation joints: {robot.data.joint_names}")
    base_ids, _ = robot.find_joints(FLOAT_BASE_JOINTS, preserve_order=True)
    base_ids = torch.as_tensor(base_ids, device=env.device)

    instruction = getattr(env_cfg, "instruction", None) or "hand_toggle"
    if isinstance(instruction, dict):
        instruction = instruction.get("default", "hand_toggle")
    cleaned_instruction = re.sub(r"[^\w\s]", "", instruction).replace(" ", "_")

    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt) if env_cfg is not None else 15

    save_sensor = save_videos and video_mode in ("all", "sensor")
    save_viewport = save_videos and video_mode in ("all", "viewport")

    video_writers_obs: list[VideoWriter] = []
    video_writers_viewport: list[VideoWriter] = []
    if save_videos:
        for env_id in range(env.num_envs):
            suffix = f"_env{env_id}" if env.num_envs > 1 else ""
            if save_sensor:
                p = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}.mp4")
                video_writers_obs.append(VideoWriter(p, video_fps))
            if save_viewport:
                p = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}_viewport.mp4")
                video_writers_viewport.append(VideoWriter(p, video_fps))

    close_hand = False
    try:
        for count in tqdm(range(num_steps)):
            if count % toggle_every == 0:
                close_hand = not close_hand
                print(f"[Step {count:04d}] Hand state: {'fist' if close_hand else 'open'}")

            wrist_pose = robot.data.joint_pos[:, base_ids]
            hand_action = torch.full((env.num_envs, 1), 1.0 if close_hand else 0.0, device=env.device)
            actions = torch.cat([wrist_pose, hand_action], dim=1)

            obs, _, term, trunc, info = env.step(actions)

            if save_videos:
                for env_id in range(env.num_envs):
                    if save_sensor:
                        frame = unpack_image_obs(obs, scale=0.5, env_id=env_id).get("combined_image")
                        if frame is not None:
                            video_writers_obs[env_id].write(frame)
                    if save_viewport:
                        frame_vp = unpack_viewport_cams(obs, env_id=env_id).get("combined_image")
                        if frame_vp is not None:
                            video_writers_viewport[env_id].write(frame_vp)
    finally:
        for vw in video_writers_obs + video_writers_viewport:
            try:
                vw.release()
            except Exception:
                pass

    return True


def main():
    output_dir = os.path.join(PACKAGE_DIR, "output", "run_sharpawave_toggle")
    os.makedirs(output_dir, exist_ok=True)

    tasks = args_cli.task or ["BananaInBowlTask"]
    auto_register_sharpa_wave_envs(task=tasks)
    task_envs = get_envs(task=tasks)
    print(f"Running SharpaWave hand toggle on {len(task_envs)} environments: {task_envs}")

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        env, env_cfg = create_env(task_env,
                                  device=args_cli.device,
                                  num_envs=args_cli.num_envs,
                                  use_fabric=True)
        try:
            print(f"Running {task_env}: '{env_cfg.instruction}'")
            run_hand_toggle_episode(
                env,
                env_cfg,
                save_videos=args_cli.save_videos,
                video_mode=args_cli.video_mode,
                num_steps=args_cli.num_steps,
                toggle_every=args_cli.toggle_every,
            )
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
