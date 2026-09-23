"""Encode timestamp-paced trial frames and write a local evidence page."""

import argparse
import html
import json
import subprocess
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()
    trial = args.trial.resolve()
    rows = [json.loads(line) for line in (trial / "events.jsonl").read_text().splitlines()]
    if not rows:
        raise ValueError("No frames to review")
    summary = json.loads((trial / "summary.json").read_text())
    config = json.loads((trial / "config.json").read_text())
    duration = (
        config["seconds"]
        if summary["stop_reason"] == "duration_complete"
        else rows[-1]["elapsed_s"] + 0.05
    )
    concat = ["ffconcat version 1.0"]
    for i, row in enumerate(rows):
        filename = row["image"]
        if (
            not filename.startswith("frames/")
            or any(c in filename for c in "'\n\r")
            or ".." in filename
        ):
            raise ValueError("Unexpected frame filename")
        end = rows[i + 1]["elapsed_s"] if i + 1 < len(rows) else duration
        concat.extend([f"file '{filename}'", f"duration {max(0.001, end - row['elapsed_s']):.6f}"])
    if (trial / "after.png").exists():
        Image.open(trial / "after.png").convert("RGB").save(trial / "after.jpg", quality=90)
        concat.append("file 'after.jpg'")
    else:
        concat.append(f"file '{rows[-1]['image']}'")
    after_image = "after.png" if (trial / "after.png").exists() else rows[-1]["image"]
    (trial / "video.ffconcat").write_text("\n".join(concat) + "\n")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-safe",
            "0",
            "-f",
            "concat",
            "-i",
            str(trial / "video.ffconcat"),
            "-vf",
            "fps=30",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(trial / "gameplay.mp4"),
        ],
        check=True,
    )
    (trial / "review.html").write_text(f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Temporal policy live trial</title>
<style>body{{max-width:1280px;margin:32px auto;padding:0 20px;background:#10151d;color:#e6edf3;font:16px system-ui}}video,img{{width:100%}}pre{{white-space:pre-wrap}}a{{color:#89c7ff}}</style>
<h1>Temporal policy live trial</h1>
<p>One checkpoint, actual screenshot feedback and bounded native controls. Video reconstructed from timestamped model-input frames, without audio or cursor.</p>
<video controls preload="metadata" src="gameplay.mp4"></video>
<p><a href="events.jsonl">Every proposal and bounded action</a> · <a href="config.json">Trial configuration</a></p>
<p>Applied steps include empty actions. First-posted-event and dispatch-start latency are reported separately when available; neither measures game response.</p>
<h2>Measurements</h2><pre>{html.escape(json.dumps(summary, indent=2))}</pre>
<h2>Before</h2><img src="before.png" alt="Game before the trial">
<h2>Last captured state</h2><img src="{html.escape(after_image)}" alt="Last captured game state">
</html>""")
    print(trial / "review.html")


if __name__ == "__main__":
    main()
