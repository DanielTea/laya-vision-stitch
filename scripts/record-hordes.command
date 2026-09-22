#!/bin/zsh
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$project_dir"
recording_dir="artifacts/human-hordes-$(date +%Y%m%d-%H%M%S)-$$"
exec .venv/bin/python -m laya_vision_stitch.demonstration_recorder record \
  --screenquest-root "$project_dir/../jev_wow_control" \
  --output "$recording_dir" "$@"
