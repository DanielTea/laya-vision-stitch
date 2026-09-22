"""Offline full-Qwen teacher qualification. No capture or game input APIs.

Reviewed menu-state instruction cases have checkable targets. Recorded next-key
cases are an ambiguous behavioral proxy, never certified expert supervision.
Development is used for teacher selection; reserved wording/games stay untouched.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .goal_curriculum import DEVELOPMENT, TRAIN
from .policy_data import check_separation, manifest_digest, read_manifest
from .trainable_model import PolicyConfig, context_text

GATES = {
    "state_accuracy_min": 0.90,
    "instruction_accuracy_min": 0.85,
    "both_goals_min": 0.80,
    "order_consistency_min": 0.90,
    "instruction_advantage_min": 0.10,
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def clean_source(source):
    return {k: copy.deepcopy(source[k]) for k in ("id", "game", "episode", "frames", "provenance")}


def prepare(data, annotations, output, seed=1729):
    review = json.loads(annotations.read_text())
    # Read the configured vocabulary instead of imposing a universal key list.
    from .d2e_data import BUTTONS

    sources = {
        s: read_manifest(data / f"{s}.jsonl", PolicyConfig(buttons=BUTTONS))
        for s in ("train", "validation")
    }
    check_separation(sources["train"], sources["validation"])
    rng = np.random.default_rng(seed)
    splits = {}
    for split, sources_split in sources.items():
        labels = {r["id"]: r for r in review["splits"][split]}
        selected = []
        for game in sorted({r["game"] for r in sources_split}):
            for opened in (False, True):
                group = [
                    r
                    for r in sources_split
                    if r["game"] == game and labels[r["id"]]["large_menu_open"] == opened
                ]
                if len(group) < 2:
                    raise ValueError("Need two examples per game/state")
                selected.extend(group[i] for i in rng.permutation(len(group))[:2])
        cases = []
        for source in selected:
            label = labels[source["id"]]
            if label["image_sha256"] != source["frames"][-1]["sha256"]:
                raise ValueError("Menu review image hash differs")
            opened = label["large_menu_open"]
            for reverse in (False, True):
                row = clean_source(source)
                row.update(
                    id=f"{source['id']}-state-{int(reverse)}",
                    source_id=source["id"],
                    task="state",
                    reverse=reverse,
                    goal="Is a large game menu overlay open in the newest screenshot? "
                    "Exclude hotbars, quest text and screens inside the 3D world.",
                    controls="",
                    previous_actions=[],
                    choices={
                        "open": "A large game menu is open.",
                        "closed": "No large game menu is open.",
                    },
                    answer="open" if opened else "closed",
                )
                if reverse:
                    row["choices"] = dict(reversed(list(row["choices"].items())))
                cases.append(row)
                for when_open in (False, True):
                    row = clean_source(source)
                    state, opposite = ("open", "closed") if when_open else ("closed", "open")
                    template = TRAIN[0] if split == "train" else DEVELOPMENT[0]
                    row.update(
                        id=f"{source['id']}-instruction-{int(reverse)}-{int(when_open)}",
                        source_id=source["id"],
                        task="instruction",
                        reverse=reverse,
                        when_open=when_open,
                        opened=opened,
                        goal=template.format(state=state, opposite=opposite),
                        controls="W is a physical keyboard button. Do not move the mouse.",
                        previous_actions=[],
                        choices={
                            "hold": "Hold W; release all other buttons.",
                            "release": "Release every button.",
                        },
                        answer="hold" if opened == when_open else "release",
                    )
                    if reverse:
                        row["choices"] = dict(reversed(list(row["choices"].items())))
                    cases.append(row)
        splits[split] = cases
    # Independent diagnostic: match a human's recorded next key set among four
    # candidates, using actual prior controls. This is not an optimality label.
    proxy = []
    for game in sorted({r["game"] for r in sources["validation"]}):
        training_sets = sorted(
            {tuple(sorted(r["action"]["buttons"])) for r in sources["train"] if r["game"] == game}
        )
        group = [r for r in sources["validation"] if r["game"] == game]
        for j in rng.permutation(len(group))[:8]:
            source = group[j]
            target = tuple(sorted(source["action"]["buttons"]))
            others = [v for v in training_sets if v != target]
            if len(others) < 3:
                raise ValueError("Need three distinct training-set distractors")
            candidates = [others[i] for i in rng.permutation(len(others))[:3]]
            position = len(proxy) % 4
            candidates.insert(position, target)
            row = clean_source(source)
            row.update(
                id=source["id"] + "-recorded",
                source_id=source["id"],
                task="recorded",
                goal=source["goal"],
                controls=source["controls"],
                previous_actions=source.get("recorded_previous_actions", []),
                choices={
                    str(i): "Hold " + ", ".join(keys) + "." if keys else "Release every button."
                    for i, keys in enumerate(candidates)
                },
                candidate_buttons={str(i): list(keys) for i, keys in enumerate(candidates)},
                answer=str(position),
            )
            proxy.append(row)
    splits["recorded"] = proxy
    check_separation(splits["train"], splits["validation"] + proxy)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in splits.items():
        (output / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    protocol = {
        "seed": seed,
        "gates": GATES,
        "source": str(data),
        "annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "digests": {s: manifest_digest(r) for s, r in splits.items()},
        "counts": {s: len(r) for s, r in splits.items()},
        "scope": "Teacher selection on reused development sessions; menu targets from reviewed "
        "state plus synthetic instructions, recorded buttons are ambiguous proxy only. "
        "No Hordes or withheld-game evaluation. No training on development rows.",
    }
    write_json(output / "protocol.json", protocol)
    return protocol


def teacher_prompt(row):
    # Explicit field allowlist: never serialize labels, review or provenance.
    labels = [chr(65 + i) for i in range(len(row["choices"]))]
    options = "\n".join(f"{k}: {v}" for k, v in zip(labels, row["choices"].values(), strict=True))
    instruction = (
        "Predict the recorded player's next physical button set over 100 ms. "
        "This is uncertain; use motion and previous controls as evidence."
        if row["task"] == "recorded"
        else "Use the newest image to answer the question or follow the instruction literally."
    )
    prompt = (
        f"{context_text(row)}\nImages are oldest to newest, with ages in seconds: "
        f"{[f['age_seconds'] for f in row['frames']]}.\n{instruction}\n{options}\n"
        "Give one short sentence describing the relevant visible evidence. "
        "Then end with a separate line FINAL: <option letter>."
    )
    return prompt, labels


def teacher(rows, output, width=640, max_tokens=128, thinking=False):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from PIL import Image

    from .backends import QWEN_ID, QWEN_REVISION, QwenVision
    from .policy_teacher import parse_final

    if output.exists():
        raise FileExistsError(output)
    qwen = QwenVision(width)
    with output.open("x") as handle:
        for index, row in enumerate(rows):
            prompt, letters = teacher_prompt(row)
            images = []
            for frame in row["frames"]:
                with Image.open(frame["image"]) as source:
                    image = source.convert("RGB")
                if image.width > width:
                    image = image.resize((width, round(image.height * width / image.width)))
                images.append(image)
            formatted = apply_chat_template(
                qwen.processor,
                qwen.model.config,
                prompt,
                num_images=len(images),
                enable_thinking=thinking,
            )
            start = time.perf_counter()
            result = generate(
                qwen.model,
                qwen.processor,
                formatted,
                image=images,
                max_tokens=max_tokens,
                temperature=0,
                verbose=False,
                **(
                    {"thinking_budget": max_tokens - 128, "enable_thinking": True}
                    if thinking
                    else {}
                ),
            )
            response = result.text
            try:
                prediction = list(row["choices"])[letters.index(parse_final(response, letters))]
            except ValueError:
                prediction = None
            record = {
                "id": row["id"],
                "prediction": prediction,
                "correct": prediction == row["answer"],
                "response": response,
                "prompt": prompt,
                "seconds": time.perf_counter() - start,
                "teacher": {
                    "model": QWEN_ID,
                    "revision": QWEN_REVISION,
                    "width": width,
                    "thinking_enabled": thinking,
                    "mode": "generated brief evidence then choice",
                    "max_tokens": max_tokens,
                    "thinking_budget": max_tokens - 128 if thinking else None,
                },
                "reviewed": False,
            }
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            print(
                f"Teacher {index + 1}/{len(rows)} {row['task']}: "
                f"correct={record['correct']} {record['seconds']:.1f}s",
                flush=True,
            )


def student(rows, bundle, output, recorded_bundle=None):
    import mlx.core as mx

    from .trainable_model import TrainableRuntime

    if output.exists():
        raise FileExistsError(output)
    runtime = TrainableRuntime.load(bundle)
    recorded_runtime = None
    records = []
    for row in rows:
        active = runtime
        if row["task"] == "recorded" and recorded_bundle is not None:
            if recorded_runtime is None:
                recorded_runtime = TrainableRuntime.load(recorded_bundle)
            active = recorded_runtime
        out = active.module(active.frames(row), *active.prepare(row))
        mx.eval(out)
        logits = np.asarray(out["buttons"][0])
        if row["task"] == "state":
            prediction = list(row["choices"])[int(mx.argmax(out["choices"][0]))]
        elif row["task"] == "instruction":
            keys = [
                k
                for k, p in zip(active.module.policy_config.buttons, logits, strict=True)
                if p >= 0
            ]
            prediction = "hold" if keys == ["w"] else "release" if not keys else "invalid"
        else:
            scores = {}
            for key, buttons in row["candidate_buttons"].items():
                targets = np.array([b in buttons for b in active.module.policy_config.buttons])
                scores[key] = float(-np.logaddexp(0, np.where(targets, -logits, logits)).sum())
            prediction = max(scores, key=scores.get)
        records.append(
            {
                "id": row["id"],
                "prediction": prediction,
                "correct": prediction == row["answer"],
                "bundle": str(recorded_bundle if active is recorded_runtime else bundle),
            }
        )
    output.write_text("".join(json.dumps(r) + "\n" for r in records))


def summarize(rows, predictions):
    by_id = {r["id"]: r for r in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r["id"] for r in rows}:
        raise ValueError("Prediction IDs must exactly match the evaluated cases")
    metrics = {}
    for task in sorted({r["task"] for r in rows}):
        selected = [r for r in rows if r["task"] == task]
        metrics[task] = {
            "n": len(selected),
            "accuracy": float(
                np.mean([by_id[r["id"]]["prediction"] == r["answer"] for r in selected])
            ),
        }
        if task in ("state", "instruction"):
            orders = {}
            for row in selected:
                orders.setdefault((row["source_id"], row.get("when_open")), []).append(row)
            if any(len(v) != 2 for v in orders.values()):
                raise ValueError("Missing reversed option case")
            metrics[task]["order_consistency"] = float(
                np.mean(
                    [
                        by_id[pair[0]["id"]]["prediction"] is not None
                        and by_id[pair[0]["id"]]["prediction"] == by_id[pair[1]["id"]]["prediction"]
                        for pair in orders.values()
                    ]
                )
            )
        if task == "instruction":
            pairs = {}
            for row in selected:
                pairs.setdefault((row["source_id"], row["reverse"]), []).append(row)
            if any(len(v) != 2 for v in pairs.values()):
                raise ValueError("Missing opposing goal case")
            metrics[task]["both_goals_correct"] = float(
                np.mean(
                    [
                        all(by_id[r["id"]]["prediction"] == r["answer"] for r in pair)
                        for pair in pairs.values()
                    ]
                )
            )
    metrics["invalid_responses"] = sum(r["prediction"] is None for r in predictions)
    return metrics


def qualify(teacher_scores, student_scores):
    t, s = teacher_scores, student_scores
    checks = {
        "state": t["state"]["accuracy"] >= GATES["state_accuracy_min"],
        "instruction": t["instruction"]["accuracy"] >= GATES["instruction_accuracy_min"],
        "both_goals": t["instruction"]["both_goals_correct"] >= GATES["both_goals_min"],
        "order_consistency": min(t[k]["order_consistency"] for k in ("state", "instruction"))
        >= GATES["order_consistency_min"],
        "advantage": t["instruction"]["accuracy"] - s["instruction"]["accuracy"]
        >= GATES["instruction_advantage_min"],
    }
    return {
        "checks": checks,
        "menu_distillation_eligible": all(checks.values()),
        "gameplay_distillation_eligible": False,
        "note": "Recorded-next-key agreement cannot certify optimal gameplay labels.",
    }


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_cases(data, split):
    protocol = json.loads((data / "protocol.json").read_text())
    if protocol["gates"] != GATES:
        raise ValueError("Qualification thresholds differ from the registered protocol")
    cases = {}
    checked = set()
    for name in ("train", "validation", "recorded"):
        cases[name] = read_rows(data / f"{name}.jsonl")
        if manifest_digest(cases[name]) != protocol["digests"][name]:
            raise ValueError("Audit manifest differs from registered protocol")
        for row in cases[name]:
            for frame in row["frames"]:
                identity = (frame["image"], frame["sha256"])
                if identity not in checked:
                    if (
                        hashlib.sha256(Path(frame["image"]).read_bytes()).hexdigest()
                        != frame["sha256"]
                    ):
                        raise ValueError("Audit image changed")
                    checked.add(identity)
    check_separation(cases["train"], cases["validation"] + cases["recorded"])
    return cases[split] + (cases["recorded"] if split == "validation" else [])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "teacher", "student", "report"))
    p.add_argument("--data", type=Path, default=Path("artifacts/teacher-audit-001"))
    p.add_argument("--source", type=Path, default=Path("artifacts/learning-gate-002"))
    p.add_argument("--annotations", type=Path, default=Path("annotations/menu-grounding-001.json"))
    p.add_argument("--bundle", type=Path, default=Path("artifacts/robust-decoder-007/bundle"))
    p.add_argument(
        "--recorded-bundle", type=Path, default=Path("artifacts/gameplay-buttons-002/bundle")
    )
    p.add_argument("--split", choices=("train", "validation"), default="validation")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--max-tokens", type=int)
    args = p.parse_args()
    if args.max_tokens is not None and (
        args.max_tokens < 1 or args.thinking and args.max_tokens <= 128
    ):
        p.error("Thinking mode must reserve at least 128 answer tokens")
    if args.mode == "prepare":
        print(json.dumps(prepare(args.source, args.annotations, args.data), indent=2))
        return
    rows = load_cases(args.data, args.split)
    teacher_name = "teacher-thinking" if args.thinking else "teacher"
    if args.mode == "teacher":
        teacher(
            rows,
            args.data / f"{teacher_name}-{args.split}.jsonl",
            max_tokens=args.max_tokens or (512 if args.thinking else 128),
            thinking=args.thinking,
        )
    elif args.mode == "student":
        student(rows, args.bundle, args.data / f"student-{args.split}.jsonl", args.recorded_bundle)
    else:
        teacher_records = read_rows(args.data / f"{teacher_name}-{args.split}.jsonl")
        student_records = read_rows(args.data / f"student-{args.split}.jsonl")
        t = summarize(rows, teacher_records)
        s = summarize(rows, student_records)
        by_game = {}
        for game in sorted({r["game"] for r in rows}):
            subset = [r for r in rows if r["game"] == game]
            ids = {r["id"] for r in subset}
            by_game[game] = {
                "teacher": summarize(subset, [r for r in teacher_records if r["id"] in ids]),
                "student": summarize(subset, [r for r in student_records if r["id"] in ids]),
            }
        report = {
            "teacher": t,
            "student": s,
            "qualification": qualify(t, s),
            "by_game": by_game,
            "teacher_configuration": teacher_records[0]["teacher"],
            "student_bundles": sorted({r["bundle"] for r in student_records}),
            "protocol": json.loads((args.data / "protocol.json").read_text()),
            "live_inputs_sent": 0,
        }
        write_json(
            args.data / f"report-{'thinking-' if args.thinking else ''}{args.split}.json", report
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
