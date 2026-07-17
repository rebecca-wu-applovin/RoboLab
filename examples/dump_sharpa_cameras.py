# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""Dump the SharpaWave GR00T policy-camera views (ego + wrist) to PNGs.

Boots one env with the sharpa GR00T registration, steps a few frames with the
hand held at its initial pose, and writes each policy camera image to
--out-dir. Fast calibration loop for camera placement — no policy server.

Usage:
    python examples/dump_sharpa_cameras.py --task BananaInBowlTask --headless \
        --out-dir /tmp/sharpa_cams
"""

import argparse
import cv2  # noqa: F401  must be imported before isaaclab
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump SharpaWave GR00T camera views.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs="+", default=["BananaInBowlTask"])
parser.add_argument("--num-steps", type=int, default=10)
parser.add_argument("--out-dir", type=str, default="output/sharpa_cams")
parser.add_argument("--franka", action="store_true",
                    help="Use the Franka+SharpaWave rig instead of the float-base hand.")
parser.add_argument("--ego-sweep", action="store_true",
                    help="Render the ego camera from a batch of candidate poses "
                         "(one sim boot) and save one PNG per candidate.")

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.registrations.sharpa_wave.auto_env_registrations_gr00t import (  # noqa: E402
    auto_register_sharpa_wave_gr00t_envs,
)
from robolab.robots.sharpa_wave import FLOAT_BASE_JOINTS  # noqa: E402


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    if args_cli.franka:
        from robolab.registrations.sharpa_wave.auto_env_registrations_franka_jointpos import (
            auto_register_franka_sharpa_envs,
        )
        auto_register_franka_sharpa_envs(task=args_cli.task)
        hold_joints = [f"panda_joint{i}" for i in range(1, 8)]
        hand_dim = 1  # binary hand action
    else:
        auto_register_sharpa_wave_gr00t_envs(task=args_cli.task)
        hold_joints = FLOAT_BASE_JOINTS
        hand_dim = 22

    task_env = get_envs(task=args_cli.task)[0]

    env, env_cfg = create_env(task_env, device=args_cli.device,
                              num_envs=args_cli.num_envs, use_fabric=True)
    robot = env.scene["robot"]
    obs, _ = env.reset()

    base_ids, _ = robot.find_joints(hold_joints, preserve_order=True)
    base_ids = torch.as_tensor(base_ids, device=env.device)

    for _ in range(args_cli.num_steps):
        wrist_pose = robot.data.joint_pos[:, base_ids]
        hand = torch.zeros((env.num_envs, hand_dim), device=env.device)
        obs, *_ = env.step(torch.cat([wrist_pose, hand], dim=1))

    def save(name, img):
        frame = img[0].detach().cpu().numpy()
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        path = os.path.join(args_cli.out_dir, f"{name}.png")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"wrote {path} ({frame.shape})")

    for name, img in obs["image_obs"].items():
        save(name, img)

    if args_cli.ego_sweep:
        from scipy.spatial.transform import Rotation

        def lookat_opengl(cam_pos, target):
            f = np.asarray(target, float) - np.asarray(cam_pos, float)
            f /= np.linalg.norm(f)
            Z = -f
            X = np.cross([0.0, 0.0, 1.0], Z); X /= np.linalg.norm(X)
            Y = np.cross(Z, X)
            q = Rotation.from_matrix(np.column_stack([X, Y, Z])).as_quat()
            return np.array([[q[3], q[0], q[1], q[2]]])  # w,x,y,z

        cam = env.scene["robot_head_camera"]
        target = np.array([0.55, 0.0, 0.0])  # table center
        candidates = {
            "a_cur_x016_z085": [0.16, -0.05, 0.85],
            "b_back_z105": [-0.10, 0.0, 1.05],
            "c_back_z120": [-0.20, 0.0, 1.20],
            "d_back_z135": [-0.30, 0.0, 1.35],
            "e_left_z100": [0.00, 0.25, 1.00],
            "f_left_z115": [-0.15, 0.35, 1.15],
        }
        for label, pos in candidates.items():
            pos_t = torch.tensor([pos], dtype=torch.float32, device=env.device) \
                + env.scene.env_origins[:1]
            quat_t = torch.tensor(lookat_opengl(pos, target), dtype=torch.float32,
                                  device=env.device)
            cam.set_world_poses(pos_t, quat_t, convention="opengl")
            for _ in range(3):
                wrist_pose = robot.data.joint_pos[:, base_ids]
                hand = torch.zeros((env.num_envs, hand_dim), device=env.device)
                obs, *_ = env.step(torch.cat([wrist_pose, hand], dim=1))
            save(f"ego_sweep_{label}", obs["image_obs"]["robot_head_camera"])

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
