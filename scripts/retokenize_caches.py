"""Re-encode cached actions in the extended vocabulary (Tab, mouse wheel, more keys).

Image tokens and goals are kept; only action tokens change. Each cached frame's action is
read from its data row, and for D2E frames mouse-wheel notches in the frame's 50 ms step
are added from the kept input logs as `scroll_up` / `scroll_down`. Writes
`<split>.tokens-extended.npz` (tokens, label_complete) and a JSON audit next to each split.
"""

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from laya_vision_stitch.p2p_adaptation import EXTENDED_KEYS, encode_action, restrict_controls

STEP = 0.05


def scroll_events(mcap):
    """[(t, dy)] mouse-wheel notches from a D2E input log (dy > 0 is up)."""
    from mcap.reader import make_reader

    events = []
    try:
        with Path(mcap).open("rb") as f:
            for _, _, message in make_reader(f).iter_messages(topics=["mouse"]):
                if b"scroll" in message.data:
                    data = json.loads(message.data)
                    if data.get("event_type") == "scroll" and data.get("dy"):
                        events.append((message.log_time / 1e9, int(data["dy"])))
    except FileNotFoundError:
        return None
    return events


def with_scroll(action, events, t):
    lo, hi = np.searchsorted(events[:, 0], [t, t + STEP]) if len(events) else (0, 0)
    window = events[lo:hi, 1] if len(events) else []
    extra = []
    if np.any(np.asarray(window) > 0):
        extra.append("scroll_up")
    if np.any(np.asarray(window) < 0):
        extra.append("scroll_down")
    return {**action, "buttons": [*action["buttons"], *extra]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--caches", nargs="+", required=True, help="cache:split")
    p.add_argument("--source", type=Path, default=Path("artifacts/d2e-source"))
    args = p.parse_args()
    pool = ProcessPoolExecutor(8)
    scrolls = {}
    for item in args.caches:
        cache, split = item.rsplit(":", 1)
        cache = Path(cache)
        rows = [
            json.loads(x) for x in (cache / f"{split}.jsonl").read_text().splitlines() if x.strip()
        ]
        data = Path(json.loads((cache / "metadata.json").read_text())["data"])
        source = {}
        for x in (data / f"{split}.jsonl").read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                source[r["id"]] = r
        recordings = {
            source[r["id"]]["source"]["recording"]
            for r in rows
            if (source[r["id"]].get("source") or {}).get("dataset") == "open-world-agents/D2E-480p"
        }
        todo = sorted(recordings - set(scrolls))
        paths = [args.source / Path(r).with_suffix(".mcap") for r in todo]
        for recording, events in zip(todo, pool.map(scroll_events, paths), strict=True):
            scrolls[recording] = None if events is None else np.array(events or np.zeros((0, 2)))
        tokens = np.zeros((len(rows), 8), np.int32)
        complete = np.zeros(len(rows), bool)
        audit = defaultdict(Counter)
        for i, r in enumerate(rows):
            data_row = source[r["id"]]
            action = data_row["action"]
            src = data_row.get("source") or {}
            events = scrolls.get(src.get("recording"))
            if events is not None:
                action = with_scroll(action, events, src["log_seconds"])
            kept, complete[i] = restrict_controls(action, EXTENDED_KEYS)
            tokens[i] = encode_action(kept, EXTENDED_KEYS)
            audit[r["game"]].update(kept["buttons"])
            audit[r["game"]]["_frames"] += 1
            audit[r["game"]]["_incomplete"] += not complete[i]
            audit[r["game"]].update(
                "dropped:" + b for b in set(action["buttons"]) - set(kept["buttons"])
            )
        np.savez(cache / f"{split}.tokens-extended.npz", tokens=tokens, label_complete=complete)
        summary = {
            game: {
                "frames": c["_frames"],
                "incomplete": c["_incomplete"],
                "scroll_steps": c["scroll_up"] + c["scroll_down"],
                "new_controls": {
                    k: v for k, v in c.items() if k in EXTENDED_KEYS[21:] or k == "tab"
                },
                "dropped": {k: v for k, v in c.items() if k.startswith("dropped:")},
            }
            for game, c in audit.items()
        }
        (cache / f"{split}.tokens-extended.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            json.dumps(
                {
                    item: {
                        g: [v["frames"], v["incomplete"], v["scroll_steps"]]
                        for g, v in summary.items()
                    }
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
