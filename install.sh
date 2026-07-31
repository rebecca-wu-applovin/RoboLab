#!/usr/bin/env bash
# One-shot RoboLab environment setup.
#
#   git clone ... && cd RoboLab && sh install.sh
#
# Installs the full IsaacSim/IsaacLab stack into a uv-managed venv. Python is
# pinned to 3.11 via .python-version (IsaacSim ships cp311 wheels only), and
# exact versions come from the committed uv.lock, so the result is reproducible
# across machines/containers.
#
# Override the Isaac stack:  ISAAC_EXTRA=isaac51 sh install.sh
set -eu

cd "$(dirname "$0")"

ISAAC_EXTRA="${ISAAC_EXTRA:-isaac50}"

# System deps (best-effort; needs root + apt): curl to bootstrap uv, ffmpeg for
# video recording.
need=""
command -v curl    >/dev/null 2>&1 || need="$need curl"
command -v ffmpeg  >/dev/null 2>&1 || need="$need ffmpeg"
command -v git-lfs >/dev/null 2>&1 || need="$need git-lfs"
if [ -n "$need" ] && command -v apt-get >/dev/null 2>&1; then
  echo ">> Installing system deps:$need"
  apt-get update -qq && apt-get install -y -qq $need \
    || echo "!! apt install skipped (need root?) — install$need manually"
fi

# Bootstrap uv if it isn't already available.
if ! command -v uv >/dev/null 2>&1; then
  echo ">> Installing uv (astral.sh)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# uv installs to ~/.local/bin (or ~/.cargo/bin on older versions) — put it on PATH.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "!! uv not on PATH after install; add \$HOME/.local/bin to PATH and re-run" >&2
  exit 1
fi

echo ">> uv sync --extra ${ISAAC_EXTRA}  (Python pinned to $(cat .python-version))"
uv sync --extra "${ISAAC_EXTRA}"

# USD scenes/objects/meshes are git-LFS; a bare clone only has pointers. Pull the
# real content so assets actually load. Everything is ~6.4 GB — narrow it with
# ROBOLAB_LFS_INCLUDE="assets/objects/taco/**,assets/fixtures/**" for a subset, or
# set ROBOLAB_SKIP_LFS=1 to skip (e.g. when assets are mounted separately).
if [ "${ROBOLAB_SKIP_LFS:-0}" != "1" ]; then
  git lfs install --local >/dev/null 2>&1 || true
  if [ -n "${ROBOLAB_LFS_INCLUDE:-}" ]; then
    echo ">> git lfs pull --include=\"${ROBOLAB_LFS_INCLUDE}\""
    git lfs pull --include="${ROBOLAB_LFS_INCLUDE}"
  else
    echo ">> git lfs pull  (all assets ~6.4 GB; ROBOLAB_SKIP_LFS=1 to skip)"
    git lfs pull
  fi
fi

cat <<'EOF'

>> RoboLab environment ready.
   Verify:  uv run pytest tests/

   For camera / RTX rendering, the CONTAINER must be launched with GPU graphics
   access — compute-only is not enough:
     * NVIDIA_DRIVER_CAPABILITIES=all   (default compute,utility will NOT render)
     * run headless with DISPLAY unset
EOF
