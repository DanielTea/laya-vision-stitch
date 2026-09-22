import copy
import json

import pytest

from laya_vision_stitch.teacher_audit import qualify, summarize, teacher_prompt


def rows():
    result = []
    for reverse in (False, True):
        result.append(
            dict(
                id=f"state-{reverse}",
                task="state",
                source_id="scene",
                reverse=reverse,
                answer="closed",
            )
        )
        for opened in (False, True):
            result.append(
                dict(
                    id=f"goal-{reverse}-{opened}",
                    task="instruction",
                    source_id="scene",
                    reverse=reverse,
                    when_open=opened,
                    answer="release" if opened else "hold",
                )
            )
    return result


def test_prompt_does_not_expose_labels_or_recorded_next_actions():
    row = dict(
        task="instruction",
        goal="Read the image",
        controls="W moves",
        previous_actions=[],
        frames=[{"age_seconds": 0}],
        choices={"hold": "Hold W", "release": "Release keys"},
        answer="SECRET_ANSWER",
        description="SECRET_STATE",
        action={"buttons": ["SECRET_BUTTON"]},
        provenance={"review": "SECRET_REVIEW"},
        opened="SECRET_OPEN",
    )
    prompt, labels = teacher_prompt(row)
    assert "SECRET" not in prompt
    assert labels == ["A", "B"]
    assert "Hold W" in prompt


def test_pair_metrics_reject_constant_answer_and_count_invalid_as_wrong():
    cases = rows()
    pred = [
        {"id": r["id"], "prediction": "closed" if r["task"] == "state" else "release"}
        for r in cases
    ]
    score = summarize(cases, pred)
    assert score["state"]["accuracy"] == 1
    assert score["instruction"]["accuracy"] == 0.5
    assert score["instruction"]["order_consistency"] == 1
    assert score["instruction"]["both_goals_correct"] == 0
    pred[0]["prediction"] = None
    score = summarize(cases, pred)
    assert score["state"]["accuracy"] == 0.5
    assert score["state"]["order_consistency"] == 0
    assert score["invalid_responses"] == 1


def test_audit_requires_complete_unique_predictions():
    cases = rows()
    pred = [{"id": r["id"], "prediction": r["answer"]} for r in cases]
    with pytest.raises(ValueError, match="exactly match"):
        summarize(cases, pred[:-1])
    with pytest.raises(ValueError, match="exactly match"):
        summarize(cases, pred + pred[:1])


def test_qualification_needs_absolute_quality_and_teacher_advantage():
    cases = rows()
    perfect = summarize(cases, [{"id": r["id"], "prediction": r["answer"]} for r in cases])
    weak = copy.deepcopy(perfect)
    weak["instruction"]["accuracy"] = 0.5
    assert qualify(perfect, weak)["menu_distillation_eligible"]
    assert not qualify(perfect, perfect)["menu_distillation_eligible"]
    assert not qualify(weak, weak)["menu_distillation_eligible"]
    assert not qualify(perfect, weak)["gameplay_distillation_eligible"]


def test_failed_qualification_prevents_target_export(tmp_path, monkeypatch):
    from laya_vision_stitch import reasoning_transfer as transfer

    cases = rows()
    monkeypatch.setattr(transfer, "load_cases", lambda *_: cases)
    for name in ("teacher", "student"):
        predictions = [{"id": r["id"], "prediction": r["answer"]} for r in cases]
        (tmp_path / f"{name}-validation.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in predictions)
        )
    target = tmp_path / "targets"
    with pytest.raises(ValueError, match="qualification failed"):
        transfer.export_targets(tmp_path, target, False)
    assert not target.exists()


def test_changed_manifest_is_rejected_before_model_loading(tmp_path):
    from laya_vision_stitch.policy_data import manifest_digest
    from laya_vision_stitch.teacher_audit import GATES, load_cases

    names = ("train", "validation", "recorded")
    for name in names:
        (tmp_path / f"{name}.jsonl").write_text("")
    (tmp_path / "protocol.json").write_text(
        json.dumps({"digests": {name: manifest_digest([]) for name in names}, "gates": GATES})
    )
    (tmp_path / "train.jsonl").write_text(json.dumps({"id": "tampered"}))
    with pytest.raises(ValueError, match="manifest differs"):
        load_cases(tmp_path, "validation")
