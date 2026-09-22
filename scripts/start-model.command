#!/bin/zsh
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$project_dir"
if [[ ! -x .venv/bin/python || ! -f artifacts/scaled-robust-001/bundle/model.safetensors ]]; then
  print 'Set up the environment and train the checkpoint described in docs/SCALING.md first.'
  exit 1
fi
exec .venv/bin/python -m laya_vision_stitch.serve --bundle artifacts/scaled-robust-001/bundle "$@"
