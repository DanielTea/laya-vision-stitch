import json

import pytest

from laya_vision_stitch.data import collect, describe, validate_splits


def test_caption_cannot_leak_logged_action_or_policy_constraints():
    state = {"health": 1, "target": "Grub", "target_signature_ok": True, "ability_ready": True}
    assert describe(state) == describe(
        {**state, "allowed": ["G"], "action": "G", "phase": "attack"}
    )
    assert "Range warning no" in describe(state)
    assert "in range" not in describe(state)


def test_collector_splits_runs_and_drops_identical_files(tmp_path):
    for i in range(8):
        run = tmp_path / f"20260921-{i:06d}-fast-live"
        run.mkdir()
        lines = []
        for j in range(13):
            path = run / f"{j}.jpg"
            path.write_bytes(b"duplicate" if j == 0 else f"image-{i}-{j}".encode())
            lines.append(
                json.dumps(
                    {
                        "image": path.name,
                        "elapsed_s": j,
                        "state": {"hud": True, "ocr_fresh": True, "health": 1},
                    }
                )
            )
        (run / "events.jsonl").write_text("\n".join(lines))
    rows = collect(tmp_path)
    assert len({r["sha256"] for r in rows}) == len(rows)
    validate_splits(rows)
    assert {r["run"] for r in rows if r["split"] == "test"} == {
        "20260921-000006-fast-live",
        "20260921-000007-fast-live",
    }


def test_split_validation_rejects_shared_run():
    with pytest.raises(ValueError, match="run crosses"):
        validate_splits(
            [
                {"run": "same", "split": "train", "sha256": "a"},
                {"run": "same", "split": "test", "sha256": "b"},
            ]
        )
