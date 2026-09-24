import json

import pytest

from scripts.closed_loop_scorecard import markdown, scorecard
from scripts.promotion_gate import main


def entry(button, onset, idle, mouse=10.0):
    return {
        "macro": {
            "button_f1": button,
            "onset_f1": onset,
            "idle_false_positive_rate": idle,
            "mouse_mae_px": mouse,
        }
    }


def report(candidate, repeat=entry(0.80, 0.0, 0.05), **extra):
    return {"frames": 10, "test": {"repeat_previous": repeat, "model": candidate, **extra}}


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def gate(tmp_path, candidate, baseline, *extra):
    output = tmp_path / "gate.json"
    code = main(
        [
            "--candidate",
            write(tmp_path, "candidate.json", candidate),
            "--baseline",
            write(tmp_path, "baseline.json", baseline),
            "--model-key",
            "model",
            "--output",
            str(output),
            *extra,
        ]
    )
    return code, json.loads(output.read_text()) if output.exists() else None


BASE = report(entry(0.85, 0.20, 0.05))


def test_candidate_passes_all_offline_gates(tmp_path):
    code, result = gate(tmp_path, report(entry(0.86, 0.25, 0.10)), BASE)
    assert code == 0 and result["passed"]
    assert {g["gate"] for g in result["gates"]} == {
        "button_f1_beats_repeat_previous",
        "onset_f1_beats_baseline",
        "idle_false_positives_within_margin",
    }


@pytest.mark.parametrize(
    "candidate,failed",
    [
        (entry(0.79, 0.25, 0.05), "button_f1_beats_repeat_previous"),
        (entry(0.86, 0.21, 0.05), "onset_f1_beats_baseline"),
        (entry(0.86, 0.25, 0.16), "idle_false_positives_within_margin"),
    ],
)
def test_each_offline_gate_can_reject(tmp_path, candidate, failed):
    code, result = gate(tmp_path, report(candidate), BASE)
    assert code == 1 and not result["passed"]
    assert [g["gate"] for g in result["gates"] if not g["passed"]] == [failed]


def test_sampled_seed_lists_are_averaged(tmp_path):
    seeds = [entry(0.82, 0.30, 0.05), entry(0.84, 0.20, 0.05)]
    code, result = gate(tmp_path, report(seeds), BASE)
    onset = next(g for g in result["gates"] if g["gate"] == "onset_f1_beats_baseline")
    assert onset["value"] == pytest.approx(0.25) and code == 0


def test_image_shuffle_control_must_degrade(tmp_path):
    good = report(entry(0.86, 0.25, 0.05), shuffled=entry(0.70, 0.1, 0.05))
    assert gate(tmp_path, good, BASE, "--shuffle-key", "shuffled")[0] == 0
    bad = report(entry(0.86, 0.25, 0.05), shuffled=entry(0.87, 0.1, 0.05))
    code, result = gate(tmp_path, bad, BASE, "--shuffle-key", "shuffled")
    assert code == 1 and not result["gates"][-1]["passed"]


def test_missing_inputs_exit_with_input_error(tmp_path):
    assert gate(tmp_path, report(entry(0.9, 0.3, 0.0)), {"validation": {}})[0] == 2
    assert gate(tmp_path, {"test": {"other": entry(0.9, 0.3, 0.0)}}, BASE)[0] == 2
    code, _ = gate(tmp_path, report(entry(0.9, 0.3, 0.0)), BASE, "--shuffle-key", "absent")
    assert code == 2


def trial(tmp_path, name="trial-001", review=None, legacy=False):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps({"bundle": "b", "seconds": 1, "execute": True})
    )
    summary = {"stop_reason": "duration_complete"}
    if review:
        summary["review"] = review
    (directory / "summary.json").write_text(json.dumps(summary))
    rows = [
        (["w", "a"], [0.0, 0.0], True, 40.0),
        ([], [0.0, 0.0], True, 50.0),
        (["tab"], [0.01, 0.0], True, 55.0),
        (["1", "mouse_left"], [0.0, 0.0], True, 60.0),
        (["2"], [0.0, 0.0], False, None),
    ]
    events = []
    for i, (buttons, mouse, applied, latency) in enumerate(rows):
        event = {
            "elapsed_s": 0.1 * (i + 1),
            "applied": applied,
            "proposal": {
                "buttons": [],
                "image_to_outputs_ms": 30.0 + i,
                "state_reset": i in (0, 4),
            },
            "bounded_action": {
                "buttons": buttons,
                "mouse_delta": mouse,
                "blocked_buttons": ["enter"] if i == 1 else [],
                "mouse_clamped": False,
            },
        }
        event["screenshot_to_post_ms" if legacy else "screenshot_to_dispatch_start_ms"] = latency
        events.append(event)
    (directory / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return directory


def test_scorecard_counts_controls_latency_and_review_signals(tmp_path):
    card = scorecard(trial(tmp_path, review={"verified_kills": 0}))
    assert card["decisions"] == 5 and card["applied_decisions"] == 4
    assert card["decisions_with_posted_events"] == 3 and card["idle_fraction"] == pytest.approx(0.4)
    assert card["control_decisions"] == {
        "movement_wasd": 1,
        "space": 0,
        "tab": 1,
        "abilities_1_4": 1,
        "mouse_buttons": 1,
        "cursor_motion": 1,
    }
    assert card["time_to_first_ability_s"] == pytest.approx(0.4)
    assert card["memory_resets"] == 2 and card["blocked_proposals"] == {"enter": 1}
    assert card["proposed_buttons"] == {}  # fixture proposals carry no buttons
    assert card["latency_ms"]["screenshot_to_dispatch_start"]["p50"] == pytest.approx(52.5)
    assert card["latency_ms"]["screenshot_to_first_event"]["samples"] == 0
    assert card["outcome_signals"]["logged_per_decision"] == []
    assert card["outcome_signals"]["manual_reviews"][0]["fields"] == {"verified_kills": 0}
    assert "trial-001" in markdown([card])


def test_scorecard_reads_legacy_dispatch_field_and_empty_trials(tmp_path):
    card = scorecard(trial(tmp_path, legacy=True))
    assert card["legacy_dispatch_field"]
    assert card["latency_ms"]["screenshot_to_dispatch_start"]["samples"] == 4
    empty = tmp_path / "empty"
    (empty / "frames").mkdir(parents=True)
    (empty / "config.json").write_text(json.dumps({"bundle": "b"}))
    assert scorecard(empty)["status"] == "no_decisions_logged"


def test_closed_loop_gates_use_scorecard(tmp_path):
    card = scorecard(trial(tmp_path))
    path = write(tmp_path, "scorecard.json", {"trials": {card["trial"]: card}})
    candidate = report(entry(0.86, 0.25, 0.05))
    passing = ["--scorecard", path, "--trial", "trial-001", "--idle-fraction-range", "0.1", "0.9"]
    assert gate(tmp_path, candidate, BASE, *passing)[0] == 0
    code, result = gate(tmp_path, candidate, BASE, *passing, "--max-latency-p50-ms", "50")
    assert code == 1
    assert [g["gate"] for g in result["gates"] if not g["passed"]] == [
        "screenshot_to_dispatch_start_p50"
    ]
    assert gate(tmp_path, candidate, BASE, "--scorecard", path)[0] == 2
