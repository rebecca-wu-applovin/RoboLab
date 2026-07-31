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

# ffmpeg is a system dep for video recording. Best-effort; needs root + apt.
if ! command -v ffmpeg >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  echo ">> Installing system ffmpeg"
  apt-get update -qq && apt-get install -y -qq ffmpeg \
    || echo "!! ffmpeg install skipped (need root?) — install it manually"
fi

echo ">> uv sync --extra ${ISAAC_EXTRA}  (Python pinned to $(cat .python-version))"
uv sync --extra "${ISAAC_EXTRA}"

cat <<'EOF'

>> RoboLab environment ready.
   Verify:  uv run pytest tests/

   For camera / RTX rendering, the CONTAINER must be launched with GPU graphics
   access — compute-only is not enough:
     * NVIDIA_DRIVER_CAPABILITIES=all   (default compute,utility will NOT render)
     * run headless with DISPLAY unset
EOF
