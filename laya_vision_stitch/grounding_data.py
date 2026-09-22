"""Reviewed real-frame questions and explicitly synthetic goal-conditioned probes."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def expand(data, annotations):
    review = json.loads(Path(annotations).read_text())
    for split in ("train", "validation"):
        rows = [
            json.loads(line) for line in (Path(data) / (split + ".jsonl")).read_text().splitlines()
        ]
        labels = {r["id"]: r for r in review["splits"][split]}
        output = []
        for source in rows:
            label = labels[source["id"]]
            if (
                hashlib.sha256(Path(source["frames"][-1]["image"]).read_bytes()).hexdigest()
                != label["image_sha256"]
            ):
                raise ValueError("Reviewed image changed")
            opened = label["large_menu_open"]
            for negative in (False, True):
                row = copy.deepcopy(source)
                for k in (
                    "action",
                    "action_chunk",
                    "recorded_previous_actions",
                    "description",
                    "teacher_probs",
                ):
                    row.pop(k, None)
                row["id"] += f"-menu-question-{int(negative)}"
                row["goal"] = (
                    "Is the game world shown without a large menu overlay?"
                    if negative
                    else "Is a large game menu overlay open?"
                )
                row["choices"] = {"yes": "Yes.", "no": "No."}
                row["answer"] = "yes" if opened != negative else "no"
                row["controls"] = ""
                row["provenance"] = {
                    **source["provenance"],
                    "label_source": "assistant-reviewed visible menu state; no player intent inference",
                }
                output.append(row)
            for forward_when_open in (False, True):
                row = copy.deepcopy(source)
                for k in (
                    "action_chunk",
                    "recorded_previous_actions",
                    "description",
                    "teacher_probs",
                ):
                    row.pop(k, None)
                row["id"] += f"-menu-instruction-{int(forward_when_open)}"
                state = "open" if forward_when_open else "closed"
                opposite = "closed" if forward_when_open else "open"
                row["goal"] = (
                    f"Hold W only while a large menu overlay is {state}; release all buttons while it is {opposite}."
                )
                row["controls"] = "W is a physical keyboard button. Do not move the mouse."
                row["choices"] = {"hold": "Hold W.", "release": "Release all buttons."}
                row["answer"] = "hold" if opened == forward_when_open else "release"
                row["action"] = {
                    "buttons": ["w"] if opened == forward_when_open else [],
                    "mouse_delta": [0.0, 0.0],
                    "pointer_xy": None,
                    "duration_seconds": 0.1,
                }
                row["provenance"] = {
                    **source["provenance"],
                    "label_source": "synthetic conditional-control probe from reviewed menu state, NOT a human demonstration",
                }
                output.append(row)
        path = Path(data) / (split + "-grounding.jsonl")
        with path.open("x") as f:
            f.writelines(json.dumps(r) + "\n" for r in output)
    return review


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--annotations", type=Path, default=Path("annotations/menu-grounding-001.json"))
    a = p.parse_args()
    expand(a.data, a.annotations)


if __name__ == "__main__":
    main()
