"""Promotion gate: explicit thresholds on offline and optional closed-loop reports.

Offline reports are shaped {split: {model_key: {"macro": {metric: value}}}}, as written by
scripts/evaluate_sequence_decoding.py; an entry that is a list (sampling seeds) is averaged
per metric. For every evaluated split the candidate entry must:
- beat the same report's `repeat_previous` macro button F1 by more than `--button-margin`;
- beat the baseline report's model macro onset F1 by more than `--onset-margin`;
- keep the idle false-positive rate within `--idle-margin` of the baseline model;
- if `--shuffle-key` is given, score above that image-shuffle control by more than
  `--shuffle-margin` (the control must degrade performance).
Closed-loop gates (with `--scorecard` and `--trial`) read closed_loop_scorecard.py output.
Exit status: 0 pass, 1 a gate failed, 2 missing or malformed inputs. Thresholds are
promotion criteria for review, not evidence of gameplay competence.
"""

import argparse
import json
import sys
from pathlib import Path

METRICS = ("button_f1", "onset_f1", "idle_false_positive_rate", "mouse_mae_px")


class GateInputError(ValueError):
    pass


def macro(report, split, key):
    try:
        entry = report[split][key]
    except (KeyError, TypeError):
        raise GateInputError(f"Missing entry {split}/{key}") from None
    entries = entry if isinstance(entry, list) else [entry]
    if not entries or not all(isinstance(e, dict) and "macro" in e for e in entries):
        raise GateInputError(f"Entry {split}/{key} has no macro metrics")
    out = {}
    for metric in METRICS:
        values = [e["macro"].get(metric) for e in entries]
        out[metric] = None if any(v is None for v in values) else sum(values) / len(values)
    out["seeds"] = len(entries)
    return out


def gate(name, split, value, threshold, passed, rule):
    if value is None or threshold is None:
        return {"gate": name, "split": split, "passed": False, "rule": rule, "detail": "missing"}
    return {
        "gate": name,
        "split": split,
        "value": value,
        "threshold": threshold,
        "passed": bool(passed),
        "rule": rule,
    }


def offline_gates(candidate, baseline, args):
    splits = args.splits or [
        s for s, v in candidate.items() if isinstance(v, dict) and args.model_key in v
    ]
    if not splits:
        raise GateInputError(f"Candidate report has no '{args.model_key}' entries")
    results = []
    for split in splits:
        cand = macro(candidate, split, args.model_key)
        repeat = macro(candidate, split, "repeat_previous")
        base = macro(baseline, split, args.baseline_model_key or args.model_key)
        threshold = repeat["button_f1"] + args.button_margin
        results.append(
            gate(
                "button_f1_beats_repeat_previous",
                split,
                cand["button_f1"],
                threshold,
                cand["button_f1"] is not None and cand["button_f1"] > threshold,
                "candidate > repeat_previous + button_margin",
            )
        )
        threshold = None if base["onset_f1"] is None else base["onset_f1"] + args.onset_margin
        results.append(
            gate(
                "onset_f1_beats_baseline",
                split,
                cand["onset_f1"],
                threshold,
                threshold is not None
                and cand["onset_f1"] is not None
                and cand["onset_f1"] > threshold,
                "candidate > baseline + onset_margin",
            )
        )
        idle = base["idle_false_positive_rate"]
        threshold = None if idle is None else idle + args.idle_margin
        value = cand["idle_false_positive_rate"]
        results.append(
            gate(
                "idle_false_positives_within_margin",
                split,
                value,
                threshold,
                threshold is not None and value is not None and value <= threshold,
                "candidate <= baseline + idle_margin",
            )
        )
        if args.shuffle_key:
            shuffled = macro(candidate, split, args.shuffle_key)[args.shuffle_metric]
            value = cand[args.shuffle_metric]
            threshold = None if shuffled is None else shuffled + args.shuffle_margin
            results.append(
                gate(
                    "image_shuffle_degrades",
                    split,
                    value,
                    threshold,
                    threshold is not None and value is not None and value > threshold,
                    f"candidate {args.shuffle_metric} > shuffled + shuffle_margin",
                )
            )
    return results


def closed_loop_gates(scorecard, args):
    try:
        card = scorecard["trials"][args.trial]
    except (KeyError, TypeError):
        raise GateInputError(f"Scorecard has no trial '{args.trial}'") from None
    if card.get("status") != "scored":
        raise GateInputError(f"Trial '{args.trial}' has no scored decisions")
    latency = card["latency_ms"][args.latency_field]["p50"]
    idle = card["idle_fraction"]
    low, high = args.idle_fraction_range
    results = [
        gate(
            "ability_presses",
            args.trial,
            card["control_decisions"]["abilities_1_4"],
            args.min_ability_presses,
            card["control_decisions"]["abilities_1_4"] >= args.min_ability_presses,
            "abilities_1_4 >= min_ability_presses",
        ),
        gate(
            "idle_fraction_bounds",
            args.trial,
            idle,
            [low, high],
            low <= idle <= high,
            "low <= idle_fraction <= high",
        ),
        gate(
            f"{args.latency_field}_p50",
            args.trial,
            latency,
            args.max_latency_p50_ms,
            latency is not None and latency < args.max_latency_p50_ms,
            "p50 < max_latency_p50_ms",
        ),
    ]
    if not args.allow_safety_stop:
        results.append(
            gate(
                "completed_without_safety_stop",
                args.trial,
                card["stop_reason"],
                "duration_complete",
                card["stop_reason"] == "duration_complete",
                "stop_reason == duration_complete",
            )
        )
    return results


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--model-key", required=True, help="candidate entry, e.g. cfg_2_greedy")
    p.add_argument("--baseline-model-key", help="baseline entry (default: --model-key)")
    p.add_argument("--splits", nargs="+")
    p.add_argument("--button-margin", type=float, default=0.0)
    p.add_argument("--onset-margin", type=float, default=0.02)
    p.add_argument("--idle-margin", type=float, default=0.10)
    p.add_argument("--shuffle-key", help="image-shuffle control entry in the candidate report")
    p.add_argument("--shuffle-metric", choices=["button_f1", "onset_f1"], default="button_f1")
    p.add_argument("--shuffle-margin", type=float, default=0.0)
    p.add_argument("--scorecard", type=Path)
    p.add_argument("--trial")
    p.add_argument("--min-ability-presses", type=int, default=1)
    p.add_argument("--idle-fraction-range", type=float, nargs=2, default=[0.0, 0.9])
    p.add_argument("--max-latency-p50-ms", type=float, default=60.0)
    p.add_argument(
        "--latency-field",
        choices=["screenshot_to_dispatch_start", "screenshot_to_first_event", "inference"],
        default="screenshot_to_dispatch_start",
    )
    p.add_argument("--allow-safety-stop", action="store_true")
    p.add_argument("--output", type=Path)
    return p


def evaluate(args):
    candidate = json.loads(args.candidate.read_text())
    baseline = json.loads(args.baseline.read_text())
    results = offline_gates(candidate, baseline, args)
    if (args.scorecard is None) != (args.trial is None):
        raise GateInputError("--scorecard and --trial must be given together")
    if args.scorecard is not None:
        results += closed_loop_gates(json.loads(args.scorecard.read_text()), args)
    return {
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "model_key": args.model_key,
        "baseline_model_key": args.baseline_model_key or args.model_key,
        "passed": all(r["passed"] for r in results),
        "gates": results,
    }


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = evaluate(args)
    except (GateInputError, OSError, json.JSONDecodeError) as error:
        print(f"promotion gate input error: {error}", file=sys.stderr)
        return 2
    for r in result["gates"]:
        status = "PASS" if r["passed"] else "FAIL"
        detail = r.get("detail") or f"{r['value']} vs {r['threshold']}"
        print(f"{status} {r['gate']} [{r['split']}] {detail}")
    print("PROMOTE" if result["passed"] else "REJECT")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
