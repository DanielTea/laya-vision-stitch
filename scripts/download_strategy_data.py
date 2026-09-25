"""Download pinned subsets of CK3 and Baldur's Gate 3 gameplay-with-input datasets.

CK3 (`Ethosoft/ck3-gameplay-mouse-keyboard-dataset`, CC-BY-4.0): sessions from the
dataset's own date split (train up to 2026-07-20, validation to 07-27, test after),
smallest-first above a minimum length until each split's budget is reached.
Baldur's Gate 3 (`yinhuankuang/rl-game-traces-baldurs-gate-3`, license "other", private
research use): whole sessions, video and input logs only (the Parquet copies are skipped).
Sessions are written under `--output/<dataset>/<session>/`; a manifest lists them.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

CK3 = ("Ethosoft/ck3-gameplay-mouse-keyboard-dataset", "95a3856c39018e49246b222935adf86b2e7122a3")
BG3 = ("yinhuankuang/rl-game-traces-baldurs-gate-3", "1bcda3bd1843cb8a0ac05ba60c63c56c7a5c8e0f")
CK3_SPLITS = {
    "train": ("ck3_20260602", "ck3_20260720_z"),
    "validation": ("ck3_20260721", "ck3_20260727_z"),
    "test": ("ck3_20260728", "ck3_20260802_z"),
}


def sessions(repo, revision):
    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
    sizes = defaultdict(dict)
    for s in info.siblings:
        if "/" in s.rfilename:
            session, name = s.rfilename.split("/", 1)
            sizes[session][name] = s.size or 0
    return sizes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/strategy-source"))
    p.add_argument(
        "--ck3-gb", nargs=3, type=float, default=[28, 4, 4], metavar=("TRAIN", "VAL", "TEST")
    )
    p.add_argument("--ck3-min-mb", type=float, default=60)
    p.add_argument("--bg3-sessions", type=int, default=8)
    args = p.parse_args()
    manifest = {"ck3": {}, "bg3": []}
    ck3 = sessions(*CK3)
    for (split, (lo, hi)), budget in zip(CK3_SPLITS.items(), args.ck3_gb, strict=True):
        names = sorted(
            (n for n in ck3 if lo <= n <= hi and sum(ck3[n].values()) >= args.ck3_min_mb * 1e6),
            key=lambda n: sum(ck3[n].values()),
        )
        chosen, total = [], 0.0
        for n in names:
            size = sum(ck3[n].values()) / 1e9
            if total + size > budget:
                continue
            chosen.append(n)
            total += size
        manifest["ck3"][split] = sorted(chosen)
        print(
            json.dumps({"ck3": split, "sessions": len(chosen), "gb": round(total, 1)}), flush=True
        )
    patterns = [f"{n}/*" for split in manifest["ck3"].values() for n in split]
    snapshot_download(
        CK3[0],
        repo_type="dataset",
        revision=CK3[1],
        allow_patterns=patterns,
        local_dir=args.output / "ck3",
    )
    print(json.dumps({"ck3": "done"}), flush=True)
    bg3 = sessions(*BG3)
    manifest["bg3"] = sorted(bg3)[: args.bg3_sessions]
    for n in manifest["bg3"]:
        snapshot_download(
            BG3[0],
            repo_type="dataset",
            revision=BG3[1],
            allow_patterns=[f"{n}/video.mkv", f"{n}/*.jsonl", f"{n}/*.json", f"{n}/*.txt"],
            local_dir=args.output / "bg3",
        )
        print(json.dumps({"bg3": n, "done": True}), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
