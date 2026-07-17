# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate GR00T N1.7's sharpa embodiment on the FR3 Duo + dual SharpaWave rig.

Server (from Isaac-GR00T, BASE model — its processor carries the
inference-ready real_r1_pro_sharpa_relative_eef embodiment):

    uv run python gr00t/eval/run_gr00t_server.py \
        --model-path nvidia/GR00T-N1.7-3B \
        --embodiment-tag REAL_R1_PRO_SHARPA \
        --device cuda --host 127.0.0.1 --port 5555 \
        --use-sim-policy-wrapper

Client:

    python policies/gr00t/run_duo.py --task BananaInBowlTask --headless
"""

import argparse
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "gr00t_duo_sharpa"

parser = argparse.ArgumentParser(description="Evaluate GR00T N1.7 sharpa embodiment on the FR3 Duo rig.")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost")
parser.add_argument("--remote-port", "--remote_port", type=int, default=5555)
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None)
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.registrations.sharpa_wave.auto_env_registrations_duo import (  # noqa: E402
    auto_register_franka_duo_envs,
)

from policies.gr00t.duo_client import GR00TDuoSharpaClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

auto_register_franka_duo_envs(
    task_dirs=args_cli.task_dirs,
    task=args_cli.task,
    ik_actions=True,
)


def make_client(args: argparse.Namespace) -> GR00TDuoSharpaClient:
    kwargs = dict(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        open_loop_horizon=args.open_loop_horizon,
    )
    return GR00TDuoSharpaClient(**{k: v for k, v in kwargs.items() if v is not None})


def main() -> None:
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
