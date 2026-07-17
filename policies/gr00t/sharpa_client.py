# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference client: GR00T N1.7 sharpa embodiment → RoboLab SharpaWave (float base).

Speaks the ``real_r1_pro_sharpa_relative_eef`` embodiment baked into the base
checkpoint (nvidia/GR00T-N1.7-3B). The embodiment was pretrained on a Galaxea
R1 Pro fitted with two Sharpa hands; RoboLab has a single right hand on a
6-DoF float base, so:

- ego view          <- over_shoulder_left_camera (exterior)
- right wrist view  <- the hand's wrist_cam
- left wrist view   <- black frames
- left arm/hand state  -> fixed neutral pose / zeros
- left arm/hand action -> ignored
- right_wrist_eef action (absolute XYZ+ROT6D after server-side decode)
    -> float-base joint targets [x, y, z, roll, pitch, yaw]
- right_hand_joints action (absolute, 22) -> finger joint targets

Known approximations, in order of expected impact: camera viewpoints don't
match R1 Pro's, the model sees a lone floating hand instead of a full robot,
the missing left side, and the unverified hand-joint ordering
(HAND_JOINTS_ORDERED in robolab/robots/sharpa_wave.py).
"""

import os
from collections import defaultdict, deque

import numpy as np
from scipy.spatial.transform import Rotation

from policies.gr00t.client import GR00TPolicyClient, resize_no_pad
from robolab.eval.base_client import InferenceClient

RESOLUTION = (240, 320)  # H, W — matches the embodiment's *_res320x240 views

# Translation from the RoboLab world frame to the R1 Pro base frame the
# embodiment was trained in. Derived from the checkpoint's statistics.json:
# training right_wrist_eef lives at x∈[0.09,0.59] (mean 0.33), y∈[−0.59,0.07]
# (mean −0.22), z∈[1.03,1.61] (mean 1.19) — a standing robot's torso frame.
# Mapping our table (center x=0.55, surface z=0) onto that distribution:
SIM_TO_R1PRO_POS = np.array([-0.22, -0.10, 1.0])

# Fixed plausible left-side state for the missing left hand: the training-mean
# right pose mirrored in y, identity-ish rotation rows, open fingers.
_LEFT_WRIST_EEF = np.array([0.33, 0.22, 1.19, 1, 0, 0, 0, 1, 0], dtype=np.float32)
_LEFT_HAND_JOINTS = np.zeros(22, dtype=np.float32)

# The float base chains roll(Rx) → pitch(Ry) → yaw(Rz) as successive local
# rotations, i.e. R = Rx@Ry@Rz — scipy's intrinsic "XYZ", not extrinsic "xyz".
_EULER_SEQ = "XYZ"


# Fixed mount rotation between the Sharpa hand-base frame (fingers +z, palm
# +x) and the R1 Pro wrist-flange EEF frame the embodiment reports: 180° about
# y. Derived by decoding the training-mean rot6d under Isaac-GR00T's ROWS
# convention (gr00t/data/state_action/pose.py::_rot6d_to_matrix): the EEF
# z-axis points back along the arm and −z extends through the fingers; with
# this mount rotation the mean pose is palm-down fingers-forward.
_HAND_TO_EEF = np.diag([-1.0, 1.0, -1.0])


def rpy_to_rot6d(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Float-base rpy -> EEF-frame 6D rotation (first two ROWS, per Isaac-GR00T)."""
    r_hand = Rotation.from_euler(_EULER_SEQ, [roll, pitch, yaw]).as_matrix()
    r_eef = r_hand @ _HAND_TO_EEF
    return r_eef[:2, :].reshape(6).astype(np.float32)


def rot6d_to_rpy(rot6d: np.ndarray) -> np.ndarray:
    """EEF-frame 6D rotation (rows) -> float-base rpy, with Gram-Schmidt cleanup."""
    r0, r1 = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    r0 = r0 / (np.linalg.norm(r0) + 1e-8)
    r1 = r1 - np.dot(r0, r1) * r0
    r1 = r1 / (np.linalg.norm(r1) + 1e-8)
    mat = np.vstack([r0, r1, np.cross(r0, r1)])
    r_hand = mat @ _HAND_TO_EEF  # its own inverse
    return Rotation.from_matrix(r_hand).as_euler(_EULER_SEQ)


def nearest_angle(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Shift target by 2πk to the representative closest to reference.

    The float-base roll/pitch/yaw joints are continuous; without this, an rpy
    flip at ±π commands a full wrist revolution.
    """
    return reference + np.arctan2(np.sin(target - reference), np.cos(target - reference))


class GR00TSharpaWaveClient(InferenceClient):
    """GR00T N1.7 sharpa-embodiment client for the SharpaWave float-base robot."""

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
        # The embodiment's video horizon is 2 frames, 1 s apart (delta_indices
        # [-20, 0] @ 20 Hz). Requests fire every open_loop_horizon env steps
        # (~0.53 s at horizon 8), so a 2-deep history per env puts the older
        # frame ~1 s back. Cold start duplicates the current frame.
        self._frame_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=2))
        print(f"[{self.__class__.__name__}] Connected; open_loop_horizon={self.open_loop_horizon}.")

    # ---- required hooks -----------------------------------------------

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        ego_image = raw_obs["image_obs"]["robot_head_camera"][env_id].clone().detach().cpu().numpy()
        wrist_image = raw_obs["image_obs"]["wrist_cam"][env_id].clone().detach().cpu().numpy()

        proprio = raw_obs["proprio_obs"]
        wrist_joints = proprio["wrist_joint_pos"][env_id].clone().detach().cpu().numpy()
        hand_joints = proprio["hand_joint_pos"][env_id].clone().detach().cpu().numpy()

        hist = self._frame_hist[env_id]
        prev = hist[0] if hist else (ego_image, wrist_image)
        hist.append((ego_image, wrist_image))

        return {
            "ego_image": ego_image,
            "wrist_image": wrist_image,
            "ego_image_prev": prev[0],
            "wrist_image_prev": prev[1],
            "wrist_joints": wrist_joints.astype(np.float64),
            "hand_joints": hand_joints.astype(np.float32),
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        def two_frames(prev_key: str, cur_key: str) -> np.ndarray:
            prev = resize_no_pad(extracted_obs[prev_key], RESOLUTION[0], RESOLUTION[1])
            cur = resize_no_pad(extracted_obs[cur_key], RESOLUTION[0], RESOLUTION[1])
            return np.stack([prev, cur])[None, ...].astype(np.uint8)  # (1, 2, H, W, C)

        ego = two_frames("ego_image_prev", "ego_image")
        wrist = two_frames("wrist_image_prev", "wrist_image")

        x, y, z, roll, pitch, yaw = extracted_obs["wrist_joints"]
        right_eef = np.concatenate([
            (np.array([x, y, z]) + SIM_TO_R1PRO_POS).astype(np.float32),
            rpy_to_rot6d(roll, pitch, yaw),
        ])

        return {
            "video.ego_view_res320x240_freq20": ego,
            "video.right_wrist_view_res320x240_freq20": wrist,
            "video.left_wrist_view_res320x240_freq20": np.zeros_like(wrist),
            "state.left_wrist_eef": _LEFT_WRIST_EEF[None, None, ...],
            "state.right_wrist_eef": right_eef[None, None, ...].astype(np.float32),
            "state.left_hand_joints": _LEFT_HAND_JOINTS[None, None, ...],
            "state.right_hand_joints": extracted_obs["hand_joints"][None, None, ...].astype(np.float32),
            "annotation.human.coarse_action": [instruction],
            # For the relative→absolute decode of the wrist action on the server.
            "_current_wrist_rpy": np.array([roll, pitch, yaw], dtype=np.float64),
        }

    def _query_server(self, request: dict) -> tuple:
        self._last_wrist_rpy = request.pop("_current_wrist_rpy")
        return self.client.get_action(request)

    def _unpack_response(self, response: tuple) -> np.ndarray:
        action_dict = response[0]

        def get(name: str) -> np.ndarray:
            for key in (f"action.{name}", name):
                if key in action_dict:
                    arr = np.asarray(action_dict[key], dtype=np.float32)
                    return arr[0] if arr.ndim == 3 else arr
            raise KeyError(f"Missing action key {name!r}; keys={sorted(action_dict)}")

        wrist_chunk = get("right_wrist_eef")      # (T, 9) absolute xyz+rot6d
        hand_chunk = get("right_hand_joints")     # (T, 22) absolute

        ref_rpy = self._last_wrist_rpy
        rows = []
        for t in range(wrist_chunk.shape[0]):
            xyz = wrist_chunk[t, :3] - SIM_TO_R1PRO_POS  # R1 Pro frame -> sim frame
            rpy = rot6d_to_rpy(wrist_chunk[t, 3:])
            rpy = nearest_angle(rpy, ref_rpy)
            ref_rpy = rpy
            rows.append(np.concatenate([xyz, rpy, hand_chunk[t]]).astype(np.float32))

        if os.environ.get("GR00T_SHARPA_DEBUG"):
            r1_xyz = wrist_chunk[[0, -1], :3]
            print(
                f"[sharpa-debug] model tgt (R1 frame) t0={np.round(r1_xyz[0], 3)} "
                f"tN={np.round(r1_xyz[-1], 3)} | sim tgt t0 xyz={np.round(rows[0][:3], 3)} "
                f"rpy={np.round(rows[0][3:6], 2)} | hand t0 mean={hand_chunk[0].mean():.2f}",
                flush=True,
            )
        return np.stack(rows)  # (T, 28)

    # ---- optional hooks -----------------------------------------------

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        ego = resize_no_pad(extracted_obs["ego_image"], RESOLUTION[0], RESOLUTION[1])
        wrist = resize_no_pad(extracted_obs["wrist_image"], RESOLUTION[0], RESOLUTION[1])
        return np.concatenate([ego, wrist], axis=1)

    def reset(self, *, env_id: int | None = None) -> None:
        super().reset(env_id=env_id)
        if env_id is None:
            self._frame_hist.clear()
        else:
            self._frame_hist.pop(env_id, None)

    def close(self) -> None:
        self.client.close()
