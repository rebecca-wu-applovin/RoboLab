# SharpaWave Hand Benchmark — Working Notes

Knowledge capture from an exploration session (2026-07-16): what RoboLab does, how to
use the `/robolab-scenegen` and `/robolab-taskgen` skills end-to-end, what was generated,
what the current devbox can and cannot run, and the plan for plugging in the SharpaWave
dexterous hand.

## What RoboLab is (one paragraph)

RoboLab is an evaluation benchmark for robot manipulation policies built on IsaacLab.
It does not train anything — it measures trained policies. A **scene** (`assets/scenes/*.usda`)
describes objects on a table; a **task** (`robolab/tasks/benchmark/*.py`) binds a scene to
language instructions (default / vague / specific) plus machine-checkable success predicates
and optional partial-credit subtasks; a **registration** (`robolab/registrations/`) combines
task × robot × cameras × action space into runnable environments; the **evaluation loop**
runs your policy as a separate server (server-client architecture, see `docs/policy.md`),
steps parallel episodes, auto-scores them, and feeds a results **dashboard** with episode
videos. Scenes and tasks are robot-agnostic — which is what makes a SharpaWave benchmark
feasible: add the hand as a robot config and all 120 existing tasks (plus custom ones) apply.

## Using `/robolab-scenegen` — walkthrough + gotchas

Flow (all steps except the last run on any machine, no IsaacSim needed):

1. **Inputs**: scene name (snake_case `.usda`), description, object count, output dir.
2. **Object selection** from `assets/objects/object_catalog.json` (312 objects; exact `name`
   values must be used).
3. **Predicates**: every object gets at least `place-on-base` (x, y anchor) + `random-rot`;
   relative predicates (`left-of`, `place-in`, ...) available. Table center (0.55, 0.0),
   safe bounds X=[0.30, 0.80], Y=[-0.40, 0.40], front = +X, left = +Y.
4. **Solve** with the predicate solver from `robolab/scene_gen/llm_scene_gen/`
   (SpatialSolver → PhysicalSolver → grammar feedback). Pure numpy + scipy.
5. **Write USDA**: `base_empty.usda` + one payload prim per object at solved (x, y, z, yaw).
6. **Settle + screenshot** via `assets/scenes/_utils/settle_scenes.py --replace --screenshot`
   — the only step that needs IsaacSim + an RTX GPU.

Gotchas learned the hard way:

- **Solver failure = objects too close.** A 0.33 m plate at table center + utensils at
  ±0.28 y fails collision resolution after 500 iterations. Fix: spread anchors toward
  table corners; the solver nudges from there (it moved our spatula y 0.36 → 0.315).
- **Duplicate catalog names exist** (two `mug` entries: `hot3d` and `ycb`). Catalog lookup
  takes the first match — the hot3d one. Check `usd_path` if you care which.
- **You don't need the full IsaacLab venv to solve placements.** A throwaway venv with just
  `numpy scipy` + `PYTHONPATH=<repo root>` runs the solver (run from the repo root — the
  catalog path is relative).
- **Payload-path depth rule**: scenes in `assets/scenes/` use `@../objects/...@`; every
  extra directory level (e.g. `assets/scenes/generated/`) needs one more `../` on both the
  object payloads and the payloads inherited from `base_empty.usda` — otherwise the table
  silently fails to load and objects settle at z ≈ −0.67 (through the table).
- The settle script's progress prints are block-buffered when redirected; run with
  `python -u` if you want live logs.

## Using `/robolab-taskgen` — summary

Inputs: scene file, instruction, episode length, output dir. The skill maps the instruction
to a success conditional ("put X in Y" → `object_in_container`, "put X on Y" →
`object_on_top`, "stack" → `stacked`, ...), always adds a `time_out` termination, uses
`require_gripper_detached=True` for placement conditions, generates vague/specific
instruction variants, subtasks (scores summing to 1.0), and attribute tags. Afterwards:
`uv run pytest tests/test_tasks_valid.py`, regenerate metadata with
`robolab/tasks/_utils/generate_task_metadata.py`, and sanity-run
`python examples/run_empty.py --task <TaskClassName>` (RTX machine required for the last one).

## Artifact generated this session

`assets/scenes/utensils_plate_dexterity.usda` — a dexterity-focused scene: `plate_small`
centered, three thin-handled utensils (`spoon` [handal], `fork_small`, `spatula_03` [vomp])
toward the table corners, `mug` [hot3d] front-right. Thin handles lying flat (8–18 mm tall)
are precision grasps a dexterous hand should beat a parallel gripper at; the mug adds a
handle-grasp/reorientation option.

**Status: solver-placed but NOT physics-settled and no screenshot** — the devbox cannot run
IsaacSim (below). On an RTX machine:

```bash
OMNI_KIT_ACCEPT_EULA=Y python assets/scenes/_utils/settle_scenes.py \
    --scene assets/scenes/utensils_plate_dexterity.usda \
    --replace --screenshot --screenshot-dir assets/scenes/_images
```

Planned companion task (not yet generated): `UtensilsOnPlateTask` — "Put the spoon, fork,
and spatula on the plate"; success = `object_on_top` × 3 with gripper detached; subtasks =
3 × `pick_and_place_on_surface` at score ⅓; 120 s; attributes `semantics`, `affordance`.

## Devbox limitation: H100s cannot run IsaacSim rendering

The current devbox has 8× H100 (driver 580.105.08). H100 is compute-only silicon — no RT
cores, no graphics engines — and RoboLab requires an RTX GPU (`README.md`, Requirements).
Verified failure chain:

- NVIDIA's Vulkan ICD refuses to initialize on H100 (`vulkaninfo`:
  `loader_scanned_icd_add: Could not get 'vkCreateInstance'`; it bails before touching
  `/dev/nvidia*`). CUDA compute is fine — torch sees all 8 GPUs.
- Kit's GPU foundation therefore fails: `omni.physx: CUDA libs are present, but no suitable
  CUDA GPU was found!` and `UsdManager: no valid foundation interface found`.
- `SimulationApp({"headless": True})` prints `app ready`, then **spins forever** in renderer
  warmup (~13 cores, 400+ threads, zero log output). It does not crash — kill it by PID.
  Two independent runs wedged at the identical point.

Practical split:

| Works on this devbox | Needs an RTX machine |
|---|---|
| Scene generation + predicate solver | `settle_scenes.py` (settle + screenshot) |
| Task-file generation | `examples/run_empty.py`, `uv run pytest tests/` |
| Metadata scripts | Any policy evaluation (camera rendering) |

The IsaacSim 5.0 stack itself is installed (`uv sync --extra isaac50` into `.venv`,
2026-07-16); imports work with `OMNI_KIT_ACCEPT_EULA=Y`.

## SharpaWave integration plan

RoboLab has no SharpaWave support today; built-in robots are Franka variants (Panda fingers
or Robotiq 2F-85, `robolab/robots/`). Per `docs/robots.md`, "if it works in IsaacLab it
works with RoboLab." Required pieces:

1. **Robot config** — a `@configclass` with a `robot: ArticulationCfg` field,
   `prim_path="{ENV_REGEX_NS}/robot"`, pointing at a SharpaWave USD (hand mounted on an
   arm; convert from URDF/MJCF if needed). Model on `robolab/robots/droid.py`. Optional
   wrist camera as a `TiledCameraCfg` field under the robot's USD hierarchy.
2. **Action config** — arm joint-position or IK actions plus the hand's finger joints.
   Biggest departure from built-ins: existing action spaces assume a 1-DoF binary gripper;
   a high-DoF hand needs its own action term and a policy client that maps model outputs
   to finger joints.
3. **`contact_gripper`** — dict of fingertip prim paths for contact sensing (single-finger
   example at `robolab/robots/droid.py:172`). Feeds grasp detection and
   `require_gripper_detached`; verify the detached logic behaves sensibly with multi-finger
   contact (`robolab/core/task/conditionals.py`) before trusting scores.
4. **Registration** — copy `robolab/registrations/example/auto_env_registration.py`, swap
   in the SharpaWave cfgs, pick camera/lighting/background variations.
5. **Policy client** — adapt one of `policies/` (pi0_family, gr00t, ...) to the hand's
   action space; the eval client interface supports batched inference.

Task strategy: start with existing pick-and-place tasks as a baseline (they run unchanged
on a new embodiment), then add dexterity-differentiating tasks via the two skills — thin
handles (HANDAL/VOMP utensils), tool use, in-hand reorientation (`reorientation` /
`affordance` attribute tags).
