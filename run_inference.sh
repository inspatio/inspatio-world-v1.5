#!/usr/bin/env bash
# Run a video (--video/--prompt/--target_traj) or a prepared scene (--scene_dir).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${INSPATIO_RUNNER_PYTHON:-python}" "$SCRIPT_DIR/run_scene_inference.py" "$@"
