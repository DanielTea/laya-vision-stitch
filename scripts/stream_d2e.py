"""Stream D2E games: download one game, import 20 FPS windows with pointer labels, delete video.

The next game downloads while the current one imports. Input logs (.mcap) are kept for
later relabeling; videos are re-downloadable at the pinned revision and are removed after
import to bound disk use. Held-out games are imported into a single `heldout` split.
"""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from laya_vision_stitch.d2e_press_frames import extract
from laya_vision_stitch.d2e_sequences import build

REPO = "open-world-agents/D2E-480p"
REVISION = "f075f7e25df6f6d385840a836f86bf92dfb877ff"


def select_recordings(game, max_bytes):
    """Recording stems spread evenly over time until the byte budget is reached."""
    from huggingface_hub import HfApi

    sizes = {}
    for entry in HfApi().list_repo_tree(
        REPO, path_in_repo=game, repo_type="dataset", revision=REVISION, recursive=True
    ):
        name = Path(entry.path)
        if name.suffix in (".mkv", ".mcap") and getattr(entry, "size", None):
            sizes[name.stem] = sizes.get(name.stem, 0) + entry.size
    stems = sorted(sizes)
    if sum(sizes.values()) <= max_bytes:
        return stems
    chosen, total = [], 0
    for fraction in np.linspace(0, 1, len(stems)):
        stem = stems[round(fraction * (len(stems) - 1))]
        if stem in chosen:
            continue
        # Interleave from both ends of the timeline so later sessions stay represented.
        order = [stem] + [stems[-1 - len(chosen)]] if len(chosen) % 2 else [stem]
        for candidate in order:
            if candidate not in chosen and total + sizes[candidate] <= max_bytes:
                chosen.append(candidate)
                total += sizes[candidate]
        if total >= max_bytes * 0.95:
            break
    return sorted(chosen)


def download(game, source, max_bytes=None):
    from huggingface_hub import snapshot_download

    patterns = [f"{game}/*"]
    if max_bytes:
        patterns = [f"{game}/{stem}.*" for stem in select_recordings(game, max_bytes)]
    snapshot_download(
        REPO,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=patterns,
        local_dir=source,
        max_workers=8,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--heldout", nargs="*", default=[])
    p.add_argument("--source", type=Path, default=Path("artifacts/d2e-source"))
    p.add_argument("--output-prefix", default="artifacts/d2e-seq")
    p.add_argument("--suffix", default="001")
    p.add_argument("--windows-per-minute", type=float, default=1.0)
    p.add_argument("--max-windows", type=int, default=600)
    p.add_argument("--heldout-max-windows", type=int, default=150)
    p.add_argument("--progress", type=Path, required=True)
    p.add_argument("--press-output", type=Path, help="Also extract click frames here")
    p.add_argument("--press-per-game", type=int, default=1500)
    p.add_argument("--press-only", action="store_true", help="Skip window import (already done)")
    p.add_argument("--max-gb-per-game", type=float, help="Download only recordings up to this size")
    args = p.parse_args()
    progress = {"repo": REPO, "revision": REVISION, "games": {}}

    def save():
        args.progress.write_text(json.dumps(progress, indent=2) + "\n")

    prefetch = {}

    def start(game):
        def work():
            try:
                download(
                    game, args.source, args.max_gb_per_game and int(args.max_gb_per_game * 1e9)
                )
                prefetch[game] = "ok"
            except Exception as error:  # noqa: BLE001 - recorded, game skipped
                prefetch[game] = f"download failed: {error!r}"

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        return thread

    threads = {args.games[0]: start(args.games[0])}
    for index, game in enumerate(args.games):
        if index + 1 < len(args.games):
            threads[args.games[index + 1]] = start(args.games[index + 1])
        threads.pop(game).join()
        entry = {"download": prefetch.get(game)}
        started = time.time()
        held = game in args.heldout
        if prefetch.get(game) == "ok" and args.press_output:
            try:
                entry["presses"] = extract(
                    args.source,
                    game,
                    args.press_output,
                    "heldout" if held else "train",
                    args.press_per_game,
                    20260923,
                )
            except Exception as error:  # noqa: BLE001 - recorded, continue
                entry["presses"] = f"failed: {error!r}"
        if prefetch.get(game) == "ok" and not args.press_only:
            output = Path(f"{args.output_prefix}-{game.lower().replace('_', '-')}-{args.suffix}")
            try:
                audit = build(
                    args.source,
                    [game],
                    output,
                    args.windows_per_minute,
                    "heldout" if held else "chronological",
                    20260923,
                    max_windows=args.heldout_max_windows if held else args.max_windows,
                )
                entry.update({"output": str(output), "frames": audit["frames"], "heldout": held})
            except Exception as error:  # noqa: BLE001 - recorded, continue with next game
                entry["import"] = f"failed: {error!r}"
        removed = 0
        for video in (args.source / game).glob("*.mkv"):
            video.unlink()
            removed += 1
        entry.update({"videos_removed": removed, "import_seconds": round(time.time() - started, 1)})
        progress["games"][game] = entry
        save()
        print(json.dumps({game: entry}), flush=True)


if __name__ == "__main__":
    main()
