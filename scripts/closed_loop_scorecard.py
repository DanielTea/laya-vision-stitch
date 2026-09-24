"""Standardized scorecards for recorded live trials; reads trial logs only, runs nothing.

Control counts use each decision's bounded (transport-filtered) action and count applied
decisions that contained the control, i.e. posted pulses, not game acknowledgement. Cursor
motion counts applied decisions with a nonzero posted mouse delta. Idle fraction is the
share of decisions that posted no input. Latencies are copied from the runner's per-decision
timestamps; the first trial's legacy `screenshot_to_post_ms` is the same dispatch-start
timestamp (docs/HORDES_TEMPORAL_LIVE.md). Outcome signals (health, XP, target health, kills)
are copied only from explicit review records, labelled with their source; decision logs are
scanned for such fields and none are inferred from pixels or actions.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

MOVEMENT = {"w", "a", "s", "d"}
ABILITIES = {"1", "2", "3", "4"}
MOUSE_BUTTONS = {"mouse_left", "mouse_right"}
OUTCOME_WORDS = ("health", "xp", "experience", "kill", "damage")
LEGACY_DISPATCH = "screenshot_to_post_ms"


def percentiles(values):
    values = [float(v) for v in values if v is not None]
    return {
        "samples": len(values),
        "p50": float(np.median(values)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
    }


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def logged_outcome_fields(events):
    fields = set()
    for event in events:
        for scope, row in (("", event), ("proposal.", event.get("proposal", {}))):
            fields |= {scope + k for k in row if any(w in k.lower() for w in OUTCOME_WORDS)}
    return sorted(fields)


def scorecard(trial, reviews=()):
    trial = Path(trial)
    config = read_json(trial / "config.json")
    if config is None:
        raise ValueError(f"{trial} has no config.json")
    summary = read_json(trial / "summary.json")
    path = trial / "events.jsonl"
    events = (
        [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
    )
    card = {
        "trial": trial.name,
        "bundle": config.get("bundle"),
        "goal": config.get("goal"),
        "capture_fps": config.get("capture_fps", 30),
        "requested_seconds": config.get("seconds"),
        "execute": config.get("execute"),
        "stop_reason": summary.get("stop_reason") if summary else None,
    }
    if not events:
        frames = trial / "frames"
        return {
            **card,
            "status": "no_decisions_logged",
            "files": sorted(p.name for p in trial.iterdir()),
            "saved_frames": len(list(frames.glob("*.jpg"))) if frames.exists() else 0,
        }
    applied = [e for e in events if e["applied"]]

    def posted(e):
        action = e["bounded_action"]
        return e["applied"] and bool(action["buttons"] or any(action["mouse_delta"]))

    presses = Counter(b for e in applied for b in e["bounded_action"]["buttons"])
    ability = [e["elapsed_s"] for e in applied if ABILITIES & set(e["bounded_action"]["buttons"])]
    elapsed = np.array([e["elapsed_s"] for e in events])
    gaps = np.diff(elapsed) * 1000
    dispatch = [e.get("screenshot_to_dispatch_start_ms", e.get(LEGACY_DISPATCH)) for e in applied]
    with_events = sum(posted(e) for e in events)
    card.update(
        {
            "status": "scored",
            "observed_span_s": float(elapsed[-1]),
            "decisions": len(events),
            "applied_decisions": len(applied),
            "decisions_with_posted_events": with_events,
            "idle_fraction": 1 - with_events / len(events),
            "decisions_per_second": (len(events) - 1) / float(elapsed[-1] - elapsed[0])
            if len(events) > 1
            else None,
            "control_decisions": {
                "movement_wasd": sum(
                    bool(MOVEMENT & set(e["bounded_action"]["buttons"])) for e in applied
                ),
                "space": presses["space"],
                "tab": presses["tab"],
                "abilities_1_4": len(ability),
                "mouse_buttons": sum(
                    bool(MOUSE_BUTTONS & set(e["bounded_action"]["buttons"])) for e in applied
                ),
                "cursor_motion": sum(any(e["bounded_action"]["mouse_delta"]) for e in applied),
            },
            "button_presses": dict(sorted(presses.items())),
            # Proposals include unapplied (observation-only / shadow) decisions.
            "proposed_buttons": dict(
                sorted(Counter(b for e in events for b in e["proposal"]["buttons"]).items())
            ),
            "blocked_proposals": dict(
                sorted(
                    Counter(
                        b for e in events for b in e["bounded_action"]["blocked_buttons"]
                    ).items()
                )
            ),
            "mouse_clamped_decisions": sum(e["bounded_action"]["mouse_clamped"] for e in events),
            "time_to_first_ability_s": min(ability) if ability else None,
            "memory_resets": sum(bool(e["proposal"].get("state_reset")) for e in events),
            "latency_ms": {
                "inference": percentiles(e["proposal"]["image_to_outputs_ms"] for e in events),
                "screenshot_to_dispatch_start": percentiles(dispatch),
                "screenshot_to_first_event": percentiles(
                    e.get("screenshot_to_first_event_ms") for e in events
                ),
                "frame_age_before_inference": percentiles(
                    e.get("frame_age_before_inference_ms") for e in events
                ),
                "decision_interval": percentiles(gaps),
            },
            "decision_gaps_over_100ms": int((gaps > 100).sum()),
            "legacy_dispatch_field": any(LEGACY_DISPATCH in e for e in events),
        }
    )
    signals = []
    if summary and summary.get("review"):
        signals.append({"source": f"{trial.name}/summary.json review", "fields": summary["review"]})
    for source, fields in reviews:
        signals.append({"source": source, "fields": fields})
    card["outcome_signals"] = {
        "logged_per_decision": logged_outcome_fields(events),
        "manual_reviews": signals,
    }
    return card


def fmt(value, digits=1):
    if value is None:
        return "–"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def markdown(cards):
    head = (
        "| Trial | Span s | Stop | Decisions | With events | Idle | WASD | Tab | 1–4 | Mouse btn "
        "| Cursor | First 1–4 s | Resets | Inference p50/p95 | Dispatch p50/p95 | First event p50/p95 |\n"
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for c in cards:
        if c["status"] != "scored":
            rows.append(f"| {c['trial']} | – | {c['status']} |" + " – |" * 13)
            continue
        k, lat = c["control_decisions"], c["latency_ms"]

        def pair(name):
            return f"{fmt(lat[name]['p50'])} / {fmt(lat[name]['p95'])}"

        rows.append(
            f"| {c['trial']} | {fmt(c['observed_span_s'])} | {c['stop_reason']} | {c['decisions']} "
            f"| {c['decisions_with_posted_events']} | {c['idle_fraction']:.2f} "
            f"| {k['movement_wasd']} | {k['tab']} | {k['abilities_1_4']} | {k['mouse_buttons']} "
            f"| {k['cursor_motion']} | {fmt(c['time_to_first_ability_s'])} | {c['memory_resets']} "
            f"| {pair('inference')} | {pair('screenshot_to_dispatch_start')} "
            f"| {pair('screenshot_to_first_event')} |"
        )
    reviews = [
        f"- **{c['trial']}** ({s['source']}): "
        + "; ".join(f"{k}: {v}" for k, v in s["fields"].items())
        for c in cards
        for s in c.get("outcome_signals", {}).get("manual_reviews", [])
    ]
    return (
        head
        + "\n".join(rows)
        + "\n\nCounts are applied decisions containing the control. Latencies in ms.\n"
        + "\nOutcome signals (manual reviews only; no per-decision outcome fields are logged):\n\n"
        + ("\n".join(reviews) if reviews else "- none")
        + "\n"
    )


def parse_review(text):
    trial, _, rest = text.partition("=")
    path, _, key = rest.partition("#")
    if not trial or not path:
        raise ValueError("Use --review TRIAL=PATH[#KEY]")
    data = json.loads(Path(path).read_text())
    return trial, (f"{path}#{key}" if key else path, data[key] if key else data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trials", type=Path, nargs="+")
    p.add_argument("--review", action="append", default=[], help="TRIAL=PATH[#KEY] manual review")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    reviews = {}
    for text in args.review:
        trial, review = parse_review(text)
        reviews.setdefault(trial, []).append(review)
    cards = [scorecard(t, reviews.get(t.name, ())) for t in args.trials]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": __doc__.split("\n\n")[1].replace("\n", " "),
        "trials": {c["trial"]: c for c in cards},
    }
    (args.output / "scorecard.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "scorecard.md").write_text(markdown(cards))
    print(markdown(cards))


if __name__ == "__main__":
    main()
