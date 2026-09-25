"""Download whole sessions (video and input logs, no Parquet copies) from `rl-game-traces-*`.

Usage: --datasets yinhuankuang/rl-game-traces-civilization-6:6 ... picks N sessions spread
evenly over each dataset's chronological list (first and last included); `repo:s1,s2` names
sessions explicitly. Downloads are pinned to the revision current at listing time, which
is recorded with the sessions in `<output>/<name>/manifest.json` (appended across calls).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, snapshot_download


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", required=True, help="repo:count")
    p.add_argument("--output", type=Path, default=Path("artifacts/strategy-source"))
    args = p.parse_args()
    for item in args.datasets:
        repo, choice = item.rsplit(":", 1)
        info = HfApi().dataset_info(repo, files_metadata=True)
        sessions = sorted({s.rfilename.split("/")[0] for s in info.siblings if "/" in s.rfilename})
        if choice.isdigit():
            spread = np.linspace(0, len(sessions) - 1, int(choice)).round()
            picks = [sessions[int(i)] for i in spread]
        else:
            picks = choice.split(",")
            missing = set(picks) - set(sessions)
            if missing:
                raise ValueError(f"Unknown sessions: {sorted(missing)}")
        name = repo.split("rl-game-traces-")[-1]
        out = args.output / name
        for session in dict.fromkeys(picks):
            snapshot_download(
                repo,
                repo_type="dataset",
                revision=info.sha,
                allow_patterns=[
                    f"{session}/video.mkv",
                    f"{session}/*.jsonl",
                    f"{session}/*.json",
                    f"{session}/*.txt",
                ],
                local_dir=out,
            )
            print(json.dumps({repo: session, "done": True}), flush=True)
        manifest = out / "manifest.json"
        previous = json.loads(manifest.read_text())["sessions"] if manifest.exists() else []
        record = {
            "repo": repo,
            "revision": info.sha,
            "sessions": list(dict.fromkeys(previous + picks)),
        }
        manifest.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
