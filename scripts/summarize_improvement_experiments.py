"""Collect compact numbers from the improvement experiments into one results JSON.

Reads ignored `artifacts/` reports and writes a tracked summary. Missing runs are
reported as absent rather than silently skipped.
"""

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ["button_f1", "onset_f1", "idle_false_positive_rate", "mouse_mae_px"]


def macro(entry):
    runs = entry if isinstance(entry, list) else [entry]
    out = {}
    for k in METRICS:
        values = [r["macro"][k] for r in runs if r["macro"].get(k) is not None]
        out[k] = round(float(np.mean(values)), 4) if values else None
    return out


def load(root, name):
    path = root / name / "report.json"
    return json.loads(path.read_text()) if path.exists() else None


def decoding(report):
    result = {}
    for split, res in report.items():
        result[split] = {
            "frames": res["frames"],
            **{k: macro(v) for k, v in res.items() if k not in ("frames", "goal_audit")},
            "goal_audit": {k: v for k, v in res["goal_audit"].items() if v is not None},
        }
    return result


def chunk(report):
    return {
        "selected_step": report["selected_step"],
        "parameters": report["parameters"],
        "splits": {
            split: {
                name: {
                    **macro(runs),
                    "predicted_controls_per_frame": round(
                        float(
                            np.mean(
                                [r["calibration"]["predicted_controls_per_frame"] for r in runs]
                            )
                        ),
                        3,
                    ),
                    "recorded_controls_per_frame": round(
                        runs[0]["calibration"]["recorded_controls_per_frame"], 3
                    ),
                }
                for name, runs in res.items()
            }
            for split, res in report["splits"].items()
        },
    }


def realtime(report):
    keys = ["button_change_rate_at_switch", "mouse_jump_at_switch", "key_flicker_rate"]
    return {
        split: {
            name: {
                **{k: macro(runs)[k] for k in ["button_f1", "onset_f1"]},
                **{
                    k: round(
                        float(
                            np.mean(
                                [r["smoothness"][k] for r in runs if r["smoothness"][k] is not None]
                            )
                        ),
                        4,
                    )
                    for k in keys
                },
            }
            for name, runs in res.items()
        }
        for split, res in report["splits"].items()
    }


def nll_conditions(report):
    return {
        split: {
            name: {
                "mean_nll": round(r["mean_nll"], 4),
                **(
                    {"post_gap_mean_nll": round(r["post_gap_mean_nll"], 4)}
                    if r.get("post_gap_mean_nll")
                    else {}
                ),
                **{k: round(r["macro"][k], 4) for k in ["button_f1", "onset_f1"]},
            }
            for name, r in res.items()
        }
        for split, res in report["splits"].items()
    }


def adapted(report):
    out = {
        "selected_step": report["selected_step"],
        "trainable_parameters": report["trainable_parameters"],
    }
    for name in report["adapted"]:
        out[name] = {
            stage: {
                "mean_nll": round(report[stage][name]["mean_nll"], 4),
                "greedy": macro(report[stage][name]["greedy"]),
                "sampled": macro(report[stage][name]["sampled"]),
            }
            for stage in ["pretrained", "adapted"]
        }
        out[name]["repeat_previous"] = macro(report["adapted"][name]["repeat_previous"])
    return out


def fovea(report):
    return {
        "source": report["source"],
        "selected_step": report["selected_step"],
        "trainable_parameters": report["trainable_parameters"],
        "splits": {
            split: {
                "mean_nll": round(r["mean_nll"], 4),
                "greedy": macro(r["greedy"]),
                "sampled": macro(r["sampled"]),
                "repeat_previous": macro(r["repeat_previous"]),
            }
            for split, r in report["splits"].items()
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--output", type=Path, default=Path("docs/improvement-experiments-results.json"))
    args = p.parse_args()
    root = args.artifacts
    groups = {
        "decoding": (
            decoding,
            [
                "seqdecode-p2p-001",
                "seqdecode-p2p-300m-001",
                "seqdecode-d2e-local-002",
                "seqdecode-d2e-local-300m-001",
            ],
        ),
        "chunk_heads": (
            chunk,
            ["chunk-flow-001", "chunk-det-001", "chunk-flow-d2e-001", "chunk-flow-p2ponly-002"],
        ),
        "realtime_chunking": (realtime, ["rtc-p2p-001", "rtc-d2e-local-002"]),
        "time_gap_memory": (nll_conditions, ["time-gap-001"]),
        "dual_system": (nll_conditions, ["dual-radio-001", "dual-efficientnet-001"]),
        "decoder_lora": (adapted, ["decoder-lora-d2e-001"]),
        "fovea": (fovea, ["fovea-fovea-001", "fovea-duplicate-001", "fovea-none-001"]),
    }
    summary = {"missing": []}
    for group, (fn, names) in groups.items():
        summary[group] = {}
        for name in names:
            report = load(root, name)
            if report is None:
                summary["missing"].append(name)
                continue
            summary[group][name] = fn(report)
    parity = load(root, "p2p-pretrained-policy-300m-001")
    if parity:
        summary["open_p2p_300m_conversion"] = {
            "max_context_abs_error": max(x["context"]["max_abs_error"] for x in parity["parity"]),
            "max_cache_abs_error": max(x["cache_max_abs_error"] for x in parity["parity"]),
            "argmax_disagreements": sum(x["argmax_disagreements"] for x in parity["parity"]),
        }
    latency = {}
    for name in ["prod-model-latency-002", "prod-model-latency-300m-001"]:
        report = load(root, name)
        if report is None:
            summary["missing"].append(name)
            continue
        latency[name] = {
            variant: {q: round(v["full_cache_ms"][q], 2) for q in ["p50", "p95"]}
            for variant, v in report["variants"].items()
        }
    pipeline = root / "prod-pipeline-002.log"
    if pipeline.exists():
        latency["prod-pipeline-002"] = {
            mode: {
                "decisions_per_second": round(v["decisions_per_second"], 2),
                **{q: round(v["screenshot_to_dispatch_start_ms"][q], 2) for q in ["p50", "p95"]},
            }
            for line in pipeline.read_text().splitlines()
            if line.startswith("{")
            for mode, v in json.loads(line).items()
        }
    else:
        summary["missing"].append("prod-pipeline-002")
    summary["idle_gpu_latency"] = latency
    # General-model round: all-game LoRA, pointer heads and the second live trial.
    general = {}
    for name in [
        "decoder-lora-all-001",
        "policy-lora-plain-001",
        "policy-lora-captions-001",
        "policy-lora-targets-001",
        "policy-lora-targets10-001",
    ]:
        report = load(root, name)
        if report is None:
            summary["missing"].append(name)
            continue
        general[name] = {
            "selected_step": report["selected_step"],
            "trainable_parameters": report["trainable_parameters"],
            "evaluations": {
                key.split("/")[-1]: {
                    stage: {
                        "mean_nll": round(report[stage][key]["mean_nll"], 4),
                        "greedy": macro(report[stage][key]["greedy"]),
                        "sampled": macro(report[stage][key]["sampled"]),
                    }
                    for stage in ["pretrained", "adapted"]
                }
                | {"repeat_previous": macro(report["adapted"][key]["repeat_previous"])}
                | (
                    {"goal_audit": report["adapted"][key]["goal_audit"]}
                    if report["adapted"][key].get("goal_audit")
                    else {}
                )
                | (
                    {"target_audit": report["adapted"][key]["target_audit"]}
                    if report["adapted"][key].get("target_audit")
                    else {}
                )
                for key in report["adapted"]
            },
        }
    for name in ["pointer-head-001", "pointer-head-002", "pointer-radio-001"]:
        report = load(root, name)
        if report is None:
            summary["missing"].append(name)
            continue
        general[name] = {
            "selected_step": report["selected_step"],
            "splits": {
                split.split("/")[-1]: {
                    k: v["macro"]
                    | {
                        g: {m: round(x, 4) for m, x in r.items()}
                        for g, r in v.items()
                        if g != "macro"
                    }
                    for k, v in res.items()
                }
                for split, res in report["splits"].items()
            },
        }
    live = root / "hordes-general-live-001" / "summary.json"
    if live.exists():
        data = json.loads(live.read_text())
        general["hordes-general-live-001"] = {
            k: data[k]
            for k in [
                "stop_reason",
                "observations",
                "applied_steps",
                "steps_with_input_events",
                "applied_buttons",
                "inference_p50_ms",
                "inference_p95_ms",
                "screenshot_to_first_event_p50_ms",
                "screenshot_to_first_event_p95_ms",
                "state_resets",
            ]
        }
    goal = load(root, "goal-pointer-001")
    if goal is None:
        summary["missing"].append("goal-pointer-001")
    else:
        general["goal-pointer-001"] = {
            "selected_step": goal["selected_step"],
            "sizes": goal["sizes"],
            **{k: goal[k] for k in ["validation", "heldout_games", "hordes_eval"] if k in goal},
        }
    # Dual system: Molmo planner + tracker with the fast controller, replayed and live;
    # tracker precision; target-conditioned controller on recorded planner answers.
    for name in [
        "planner-replay-004",
        "hordes-planner-live-001",
        "target-replay-general-001",
        "target-replay-noplanner-001",
        "hordes-act-live-001",
        "hordes-act-live-002",
    ]:
        path = root / name / "summary.json"
        if path.exists():
            general[name] = json.loads(path.read_text())
        else:
            summary["missing"].append(name)
    tracker = root / "tracker-eval-001" / "report.json"
    if tracker.exists():
        general["tracker-eval-001"] = json.loads(tracker.read_text())
    else:
        summary["missing"].append("tracker-eval-001")
    summary["general_round"] = general
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"missing": summary["missing"]}))


if __name__ == "__main__":
    main()
