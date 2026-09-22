"""Paired conditional instructions with disjoint wording families and label isolation."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

TRAIN = (
    "Hold W if a large game menu is {state}. Otherwise release every key.",
    "When the large menu overlay is {state}, keep W down. When it is {opposite}, stop pressing buttons.",
    "Press W only with the menu {state}; keep all keys released in the other case.",
    "Your goal is to hold W while the menu is {state}, and release it while the menu is {opposite}.",
    "If you see that the large menu is {state}, use W. Otherwise do not press any button.",
    "Keep the W key pressed exactly when the game menu is {state}. Release all controls otherwise.",
    "Hold W only while a large menu overlay is {state}; release all buttons while it is {opposite}.",
)
DEVELOPMENT = (
    "Release all keys unless the large menu is {state}; in that case hold W.",
    "Check the screenshot: W should be down for a {state} menu and up for a {opposite} menu.",
)
SEALED = (
    "Use the visible menu to decide whether to press W: {state} means press, {opposite} means release.",
    "Keep your hands off the keys whenever the menu is {opposite}. Only keep W held when it is {state}.",
)


INVERSE_TRAIN = (
    "Release every key when the menu is {opposite}. Hold W when the menu is {state}.",
    "If the game menu is {opposite}, let go of W. Otherwise keep W held down.",
    "Do not hold W while the large menu is {opposite}. Press W in the other menu state.",
    "Keep every key released while the menu is {opposite}; otherwise keep W down.",
    "A {opposite} menu means release all keys; a {state} menu means W held.",
    "Avoid pressing W with the menu {opposite}. Press it with the menu {state}.",
    "When the menu is {opposite}, the required action is release; when it is {state}, the required action is W.",
)
SEALED_V2 = (
    "Let W stay down only in the {state} menu state. In the {opposite} state, W must be up.",
    "If the menu is {opposite}, do not press anything. In all other cases keep pressing W.",
)


LOGIC_TRAIN = (
    "Hold W unless the menu is {opposite}; if it is {opposite}, release every key.",
    "Release W unless the game menu is {state}; if it is {state}, hold W.",
    "When the menu is not {opposite}, hold W. Otherwise release every key.",
    "When the menu is not {state}, release W. Otherwise hold the W key.",
    "Keep your fingers off W while the menu is {opposite}; keep W pressed while it is {state}.",
    "Leave the keyboard alone if the menu is {opposite}. Use W if it is {state}.",
    "All keys must be up with a {opposite} menu. With a {state} menu, W must be down.",
    "W must be down with a {state} menu. All keys must be up with a {opposite} menu.",
    "Do not press any key unless the menu is {state}. In that state press W.",
    "Do not release W unless the menu is {opposite}. In that state release all keys.",
    "Press W in every case except a {opposite} menu. For that exception release all buttons.",
    "Release all buttons in every case except a {state} menu. For that exception hold W.",
)


def expand(rows, labels, templates):
    result = []
    for source in rows:
        label = labels[source["id"]]
        if (
            hashlib.sha256(Path(source["frames"][-1]["image"]).read_bytes()).hexdigest()
            != label["image_sha256"]
        ):
            raise ValueError("Reviewed image hash differs")
        opened = label["large_menu_open"]
        for template_id, template in templates:
            for reverse in (False, True):
                for when_open in (False, True):
                    # Explicit whitelist: demonstration actions, descriptions,
                    # answer targets and previous-action labels cannot leak.
                    row = {
                        k: copy.deepcopy(source[k])
                        for k in ("game", "episode", "frames", "provenance")
                    }
                    row["id"] = f"{source['id']}-goal-{template_id}-{int(reverse)}-{int(when_open)}"
                    state, opposite = ("open", "closed") if when_open else ("closed", "open")
                    row["goal"] = template.format(state=state, opposite=opposite)
                    row["controls"] = "W is a physical keyboard button. Do not move the mouse."
                    choices = {"act": "Take the next action.", "wait": "Wait."}
                    row["choices"] = dict(reversed(list(choices.items()))) if reverse else choices
                    row["previous_actions"] = []
                    row["action"] = {
                        "buttons": ["w"] if opened == when_open else [],
                        "mouse_delta": [0.0, 0.0],
                        "pointer_xy": None,
                        "duration_seconds": 0.1,
                    }
                    row["_sampling_stratum"] = [int(opened), int(when_open)]
                    row["_pair_variant"] = f"{template_id}-{int(reverse)}"
                    row["template_id"] = template_id
                    row["options_reversed"] = reverse
                    row["provenance"]["label_source"] = (
                        "synthetic instruction from assistant-reviewed menu state; not human intent"
                    )
                    result.append(row)
    return result


def create(data, annotations, output, version=1):
    output.mkdir(parents=True, exist_ok=False)
    review = json.loads(annotations.read_text())
    sources = {
        s: [json.loads(x) for x in (data / f"{s}.jsonl").read_text().splitlines()]
        for s in ("train", "validation")
    }
    train_text = TRAIN if version == 1 else TRAIN + INVERSE_TRAIN
    if version == 3:
        train_text += LOGIC_TRAIN
    train_templates = [(f"train-{i}", t) for i, t in enumerate(train_text)]
    validation_templates = [("seen-0", TRAIN[0])] + [
        (f"dev-{i}", t) for i, t in enumerate(DEVELOPMENT)
    ]
    if version >= 2:
        validation_templates += [(f"previous-sealed-{i}", t) for i, t in enumerate(SEALED)]
    sealed_text = SEALED if version == 1 else SEALED_V2
    sealed_templates = [(f"sealed-{i}", t) for i, t in enumerate(sealed_text)]
    for split, source_split, templates in [
        ("train", "train", train_templates),
        ("validation", "validation", validation_templates),
        ("sealed", "validation", sealed_templates),
    ]:
        rows = expand(
            sources[source_split], {r["id"]: r for r in review["splits"][source_split]}, templates
        )
        (output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    protocol = {
        "training_templates": train_text,
        "version": version,
        "previously_consumed_wordings": SEALED if version >= 2 else [],
        "development_templates": DEVELOPMENT,
        "sealed_templates": sealed_text,
        "source": str(data),
        "annotations": str(annotations),
        "sealed_scope": "new wording only; same reused development scenes, not untouched games",
        "choices": "generic act/wait, both orders; no semantic action options",
        "labels": "assistant-reviewed visible menu state, synthetic paired goals",
    }
    (output / "curriculum.json").write_text(json.dumps(protocol, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("artifacts/learning-gate-002"))
    p.add_argument("--annotations", type=Path, default=Path("annotations/menu-grounding-001.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--version", type=int, choices=[1, 2, 3], default=1)
    args = p.parse_args()
    create(args.data, args.annotations, args.output, args.version)


if __name__ == "__main__":
    main()
