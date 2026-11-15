#!/usr/bin/env bash
# Explicit entrypoint for the operator; does not install, provision, or shut down anything.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${TEXT_CONTROLS_PYTHON:-python}"
"$PYTHON_BIN" -m unittest tests.test_text_controls
"$PYTHON_BIN" scripts/run_text_controls.py "$@"
