# Changelog

## [0.2.1] - 2026-07-20

### Added

- Faithful replay of recorded episodes: new `robolab/core/replay/` module with
  recorded env-config overlay, initial-state restore, and per-step state
  validation; driver script `examples/run_recorded.py`; user guide
  `docs/replay.md`. ([#19](https://github.com/NVlabs/RoboLab/issues/19))
- Batched inference support in the eval client interface.
- The pytest install-verification suite now ships in `tests/` (documented
  fresh-install check: `uv run pytest tests/`).

### Changed

- Subtask progress tracking is enabled by default during evaluation.
- Updated the GR00T N1.7 client.
  ([#15](https://github.com/NVlabs/RoboLab/pull/15))
- Overrode isaacsim's stale `websockets==12.0` pin so `uv lock` resolves a
  modern websockets for the policy clients.

### Fixed

- `WorldState` extras handling on IsaacLab 2.3 (`XformPrimView`).

## [0.2.0] - 2026-07-07

### Added

- IsaacSim 5.1 / IsaacLab 2.3 support alongside 5.0 / 2.2, selected via the
  mutually-exclusive `isaac50` / `isaac51` extras.
- Relative differential-IK environment registrations
  (`examples/run_rel_ik_demo.py`).
- Realtime/pathtracing RTX renderer selection via `--rendering-type`.
- Poly Haven attribution files for the curated backgrounds.

### Changed

- Bundled backgrounds trimmed to a curated indoor set. More backgrounds can be obtained from Poly Haven.
- End-effector pose position is recorded in the env-local frame.
- Docs: asset contribution guidelines, conditionals
  restructure; generalized VRAM sizing guide.

### Fixed

- HDF5 episode success is derived from `terminated` rather than a named
  termination term.

## [0.1.1] - 2026-05-31

### Added

- Cosmos3 policy client; refreshed DreamZero, GR00T, and Pi0-family clients.
- Results dashboard (`robolab-dashboard` CLI).
- `/robolab-scenegen` and `/robolab-taskgen` Claude Code skills.
- Convex-hull placement predicates, adaptive sampling, and DROID IK control.
- `CONTRIBUTING.md` and a `uv`-based install flow.

### Changed

- Per-policy reorganization: one runner per backend under
  `policies/<policy>/run.py`.
- Docs revamp: per-policy READMEs, SPDX headers, `THIRD_PARTY_NOTICES.md`.
- Episode videos encoded as streaming H.264 (libx264) so they play in
  browsers.

### Fixed

- Placement order enforced in then-sequenced tasks.
- Frozen-env data leaks in the episode recorder.

## [0.1.0] - 2026-04-10

Initial public release: 100+ benchmark tasks,
DROID and Franka embodiments, server-client policy evaluation (Pi0 family,
GR00T, DreamZero), and HDF5 episode recording with analysis tools.

[0.2.1]: https://github.com/NVlabs/RoboLab/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/NVlabs/RoboLab/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/NVlabs/RoboLab/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/NVlabs/RoboLab/releases/tag/v0.1.0
