#!/usr/bin/env bash
# Run the six bundled example scenes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${INSPATIO_RUNNER_PYTHON:-python}" "$SCRIPT_DIR/run_scene_inference.py" "$@"
