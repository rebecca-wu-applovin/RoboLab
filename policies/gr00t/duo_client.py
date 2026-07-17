# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7 sharpa embodiment → FR3 Duo + dual SharpaWave rig.

Much closer to the pretraining embodiment (Galaxea R1 Pro: dual arms, two
Sharpa hands, head + two wrist cameras) than the floating-hand client:
both wrist views are real, both hands report state, and the arms are driven
through absolute-pose differential IK (FrankaDuoSharpaIKActionCfg,
7+7+22+22 = 58-dim actions).

Frame conventions (see sharpa_client.py for derivations):
- rot6d is first-two-ROWS (Isaac-GR00T pose.py).
- EEF frame = wrist flange with _HAND_TO_EEF = diag(-1, 1, -1) vs the hand
  base; our IK body is panda_link8, hand base = link8 ⊗ Rz(-45°).
- R1 base frame = env frame + (0, 0, 1.0) (their wrists live at z≈1.0-1.6,
  ours at z≈0.0-0.6 over the table).
"""

import os
from collections import defaultdict, deque

import numpy as np
from scipy.spatial.transform import Rotation

from policies.gr00t.client import GR00TPolicyClient, resize_no_pad
from policies.gr00t.sharpa_client import _EULER_SEQ  # noqa: F401  (doc parity)
from robolab.eval.base_client import InferenceClient

RESOLUTION = (240, 320)  # H, W

SIM_TO_R1_POS = np.array([0.0, 0.0, 1.0])
# Robot root (torso) in the env frame — IK pose commands are in the root frame.
ROBOT_ROOT_POS = np.array([-0.35, 0.0, 0.45])

_HAND_TO_EEF = np.diag([-1.0, 1.0, -1.0])
_RZ_M45 = Rotation.from_euler("z", -45, degrees=True).as_matrix()  # link8 -> hand flange


def _rot6d_rows_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    r0, r1 = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    r0 = r0 / (np.linalg.norm(r0) + 1e-8)
    r1 = r1 - np.dot(r0, r1) * r0
    r1 = r1 / (np.linalg.norm(r1) + 1e-8)
    return np.vstack([r0, r1, np.cross(r0, r1)])


def _matrix_to_rot6d_rows(mat: np.ndarray) -> np.ndarray:
    return mat[:2, :].reshape(6).astype(np.float32)


def _flange_state_9d(pos_env: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Env-frame flange pose -> 9D EEF state in the R1 frame."""
    w, x, y, z = quat_wxyz
    r_link8 = Rotation.from_quat([x, y, z, w]).as_matrix()
    r_hand = r_link8 @ _RZ_M45
    r_eef = r_hand @ _HAND_TO_EEF
    p = np.asarray(pos_env, dtype=np.float64) + SIM_TO_R1_POS
    return np.concatenate([p, _matrix_to_rot6d_rows(r_eef)]).astype(np.float32)


def _eef_action_to_ik_cmd(eef9d: np.ndarray) -> np.ndarray:
    """Absolute 9D EEF target (R1 frame) -> IK pose command (root frame, wxyz)."""
    p_env = np.asarray(eef9d[:3], dtype=np.float64) - SIM_TO_R1_POS
    p_root = p_env - ROBOT_ROOT_POS
    r_eef = _rot6d_rows_to_matrix(eef9d[3:])
    r_hand = r_eef @ _HAND_TO_EEF  # its own inverse
    r_link8 = r_hand @ _RZ_M45.T
    q = Rotation.from_matrix(r_link8).as_quat()  # xyzw
    return np.concatenate([p_root, [q[3], q[0], q[1], q[2]]]).astype(np.float32)


class GR00TDuoSharpaClient(InferenceClient):
    """GR00T N1.7 sharpa-embodiment client for the FR3 Duo + dual SharpaWave rig."""

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 5555,
        open_loop_horizon: int = 8,
        api_token: str | None = None,
    ) -> None:
        super().__init__()
        self.open_loop_horizon = int(open_loop_horizon)
        print(f"[{self.__class__.__name__}] Connecting to GR00T policy server at {remote_host}:{remote_port}...")
        self.client = GR00TPolicyClient(host=remote_host, port=remote_port, api_token=api_token)
        # video horizon is 2 frames ~1 s apart; requests fire every
        # open_loop_horizon env steps (~0.53 s at horizon 8)
        self._frame_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=2))
        print(f"[{self.__class__.__name__}] Connected; open_loop_horizon={self.open_loop_horizon}.")

    # ---- required hooks -----------------------------------------------

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        imgs = raw_obs["image_obs"]
        ego = imgs["robot_head_camera"][env_id].clone().detach().cpu().numpy()
        left_w = imgs["left_wrist_cam"][env_id].clone().detach().cpu().numpy()
        right_w = imgs["right_wrist_cam"][env_id].clone().detach().cpu().numpy()

        prop = raw_obs["proprio_obs"]
        get = lambda k: prop[k][env_id].clone().detach().cpu().numpy()

        hist = self._frame_hist[env_id]
        prev = hist[0] if hist else (ego, left_w, right_w)
        hist.append((ego, left_w, right_w))

        return {
            "ego": ego, "left_w": left_w, "right_w": right_w,
            "ego_prev": prev[0], "left_w_prev": prev[1], "right_w_prev": prev[2],
            "left_eef": _flange_state_9d(get("left_eef_pos"), get("left_eef_quat")),
            "right_eef": _flange_state_9d(get("right_eef_pos"), get("right_eef_quat")),
            "left_hand": get("left_hand_joint_pos").astype(np.float32),
            "right_hand": get("right_hand_joint_pos").astype(np.float32),
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        def two(prev_key, cur_key):
            prev = resize_no_pad(extracted_obs[prev_key], RESOLUTION[0], RESOLUTION[1])
            cur = resize_no_pad(extracted_obs[cur_key], RESOLUTION[0], RESOLUTION[1])
            return np.stack([prev, cur])[None, ...].astype(np.uint8)

        return {
            "video.ego_view_res320x240_freq20": two("ego_prev", "ego"),
            "video.left_wrist_view_res320x240_freq20": two("left_w_prev", "left_w"),
            "video.right_wrist_view_res320x240_freq20": two("right_w_prev", "right_w"),
            "state.left_wrist_eef": extracted_obs["left_eef"][None, None, ...],
            "state.right_wrist_eef": extracted_obs["right_eef"][None, None, ...],
            "state.left_hand_joints": extracted_obs["left_hand"][None, None, ...],
            "state.right_hand_joints": extracted_obs["right_hand"][None, None, ...],
            "annotation.human.coarse_action": [instruction],
        }

    def _query_server(self, request: dict) -> tuple:
        return self.client.get_action(request)

    def _unpack_response(self, response: tuple) -> np.ndarray:
        action_dict = response[0]

        def get(name):
            for key in (f"action.{name}", name):
                if key in action_dict:
                    arr = np.asarray(action_dict[key], dtype=np.float32)
                    return arr[0] if arr.ndim == 3 else arr
            raise KeyError(f"Missing action key {name!r}; keys={sorted(action_dict)}")

        left_eef = get("left_wrist_eef")     # (T, 9) absolute, R1 frame
        right_eef = get("right_wrist_eef")   # (T, 9)
        left_hand = get("left_hand_joints")  # (T, 22)
        right_hand = get("right_hand_joints")

        rows = []
        for t in range(left_eef.shape[0]):
            rows.append(np.concatenate([
                _eef_action_to_ik_cmd(left_eef[t]),
                _eef_action_to_ik_cmd(right_eef[t]),
                left_hand[t],
                right_hand[t],
            ]).astype(np.float32))

        if os.environ.get("GR00T_SHARPA_DEBUG"):
            r = rows[0]
            print(f"[duo-debug] L tgt root xyz={np.round(r[:3],3)} R tgt root xyz={np.round(r[7:10],3)} "
                  f"| Rhand mean={right_hand[0].mean():.2f} Lhand mean={left_hand[0].mean():.2f}", flush=True)
        return np.stack(rows)  # (T, 58)

    # ---- optional hooks -----------------------------------------------

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        panels = [resize_no_pad(extracted_obs[k], RESOLUTION[0], RESOLUTION[1])
                  for k in ("ego", "left_w", "right_w")]
        return np.concatenate(panels, axis=1)

    def reset(self, *, env_id: int | None = None) -> None:
        super().reset(env_id=env_id)
        if env_id is None:
            self._frame_hist.clear()
        else:
            self._frame_hist.pop(env_id, None)

    def close(self) -> None:
        self.client.close()
