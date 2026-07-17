# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""Sweep candidate FR3 Duo arm ready-poses and dump ego + viewport frames.

One sim boot; for each candidate the arms are position-driven to the pose,
settled for a few steps, and the ego/viewport views saved as PNGs.

    python examples/dump_duo_poses.py --headless --out-dir output/duo_poses
"""

import argparse
import cv2  # noqa: F401  must be imported before isaaclab
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep FR3 Duo ready poses.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs="+", default=["BananaInBowlTask"])
parser.add_argument("--settle-steps", type=int, default=25)
parser.add_argument("--out-dir", type=str, default="output/duo_poses")

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.registrations.sharpa_wave.auto_env_registrations_duo import (  # noqa: E402
    auto_register_franka_duo_envs,
)

# Right-arm candidates (j1..j7); left arm mirrors with sign flips on j1/j3/j5/j7.
CANDIDATES = {
    "g_fold": [0.0, 0.3, 0.0, -2.2, 0.0, 2.0, 0.0],
    "h_fold_out": [0.4, 0.5, 0.0, -2.2, 0.0, 2.4, 0.0],
    "i_fold_reach": [0.0, 0.7, 0.0, -2.0, 0.0, 2.4, 0.0],
    "j_deep_fold": [0.6, 0.6, -0.3, -2.4, 0.0, 2.6, 0.0],
    "k_low_hover": [0.2, 0.9, 0.0, -1.9, 0.0, 2.6, 0.0],
    "l_wide": [0.9, 0.4, -0.4, -2.1, 0.2, 2.3, 0.0],
}

MIRROR = np.array([-1, 1, -1, 1, -1, 1, -1], dtype=np.float32)


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    auto_register_franka_duo_envs(task=args_cli.task)
    task_env = get_envs(task=args_cli.task)[0]

    env, env_cfg = create_env(task_env, device=args_cli.device,
                              num_envs=args_cli.num_envs, use_fabric=True)
    obs, _ = env.reset()

    def save(name, img):
        frame = img[0].detach().cpu().numpy()
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        path = os.path.join(args_cli.out_dir, f"{name}.png")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"wrote {path}")

    for label, right in CANDIDATES.items():
        right_t = torch.tensor([right], dtype=torch.float32, device=env.device)
        left_t = torch.tensor([np.asarray(right, dtype=np.float32) * MIRROR],
                              dtype=torch.float32, device=env.device)
        hands = torch.zeros((env.num_envs, 2), device=env.device)
        actions = torch.cat([left_t, right_t, hands], dim=1)
        for _ in range(args_cli.settle_steps):
            obs, *_ = env.step(actions)
        save(f"pose_{label}_ego", obs["image_obs"]["robot_head_camera"])
        save(f"pose_{label}_viewport", obs["viewport_cam"]["egocentric_mirrored_camera"])

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
