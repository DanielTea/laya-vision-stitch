"""Add cursor/click labels to existing D2E sequence manifests from their kept input logs.

Writes new manifests (same frame files) into a new directory; the source is unchanged.
"""

import argparse
import json
from pathlib import Path

from laya_vision_stitch.d2e_pointer import PointerTimeline
from laya_vision_stitch.d2e_sequences import STEP


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path("artifacts/d2e-source"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    timelines, counts = {}, {}
    for manifest in sorted(args.data.glob("*.jsonl")):
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        for r in rows:
            recording = r["source"]["recording"]
            if recording not in timelines:
                mcap = (args.source / recording).with_suffix(".mcap")
                try:
                    timelines[recording] = PointerTimeline.from_mcap(mcap)
                except (ValueError, FileNotFoundError):
                    timelines[recording] = None
            timeline = timelines[recording]
            r["pointer"] = (
                None if timeline is None else timeline.label(r["source"]["log_seconds"], STEP)
            )
        (args.output / manifest.name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        counts[manifest.stem] = {
            "frames": len(rows),
            "with_pointer": sum(r["pointer"] is not None for r in rows),
            "presses": sum(bool(r["pointer"] and r["pointer"]["press"]) for r in rows),
        }
    (args.output / "audit.json").write_text(
        json.dumps({"source": str(args.data), "counts": counts}, indent=2) + "\n"
    )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
