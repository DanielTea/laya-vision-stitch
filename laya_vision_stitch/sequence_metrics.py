"""Sequence-aware action metrics, including onsets that persistence baselines cannot score.

Button F1 rewards holding whatever was held before, so repeating the previous action
looks strong. Onset F1 scores only newly pressed controls, matched within a tolerance.
"""

from collections import defaultdict

import numpy as np

from .p2p_adaptation import KEYS_WITH_TAB
from .p2p_pretrained_policy import MOUSE_NAMES, MOUSE_X, MOUSE_Y

BUTTONS = tuple(dict.fromkeys(k for k in (*KEYS_WITH_TAB, *MOUSE_NAMES) if k is not None))


def token_buttons(tokens):
    tokens = np.asarray(tokens)
    result = []
    for row in tokens.reshape(-1, 8):
        names = {KEYS_WITH_TAB[t] for t in row[:4]} | {MOUSE_NAMES[t] for t in row[4:6]}
        result.append(frozenset(n for n in names if n is not None))
    return result


def token_mouse(tokens):
    tokens = np.asarray(tokens).reshape(-1, 8)
    return np.stack([np.asarray(MOUSE_X)[tokens[:, 6]], np.asarray(MOUSE_Y)[tokens[:, 7]]], 1)


def _f1(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


def onsets(buttons, previous):
    """Controls pressed now that were not held in the recorded previous step."""
    return [set(b) - set(p) for b, p in zip(buttons, previous, strict=True)]


def evaluate_actions(rows, predicted, truth, pred_mouse=None, true_mouse=None, tolerance=2):
    """rows need game/sequence/step; predicted/truth are lists of button sets per frame.

    Teacher-forced: a predicted onset is a control absent from the recorded previous frame.
    Onsets match the same control within +/- tolerance frames in the same sequence.
    """
    order = defaultdict(list)
    for i, r in enumerate(rows):
        order[r["sequence"]].append(i)
    previous = [frozenset()] * len(rows)
    first = set()
    for idx in order.values():
        idx.sort(key=lambda i: rows[i]["step"])
        first.add(idx[0])
        for a, b in zip(idx[:-1], idx[1:], strict=True):
            previous[b] = truth[a]
    games = defaultdict(lambda: defaultdict(float))
    for i, r in enumerate(rows):
        m = games[r["game"]]
        got, want = set(predicted[i]), set(truth[i])
        m["tp"] += len(got & want)
        m["fp"] += len(got - want)
        m["fn"] += len(want - got)
        m["exact"] += got == want
        m["examples"] += 1
        m["idle"] += not want
        m["idle_fp"] += (not want) and bool(got)
        if pred_mouse is not None:
            moving = np.any(np.asarray(true_mouse[i]) != 0)
            if moving:
                m["mouse_error"] += float(np.abs(np.asarray(pred_mouse[i]) - true_mouse[i]).mean())
                m["moving"] += 1
    # Onset matching per sequence and control.
    pred_on, true_on = onsets(predicted, previous), onsets(truth, previous)
    for idx in order.values():
        game = rows[idx[0]]["game"]
        m = games[game]
        valid = [i for i in idx if i not in first]
        for button in BUTTONS:
            p = [rows[i]["step"] for i in valid if button in pred_on[i]]
            t = [rows[i]["step"] for i in valid if button in true_on[i]]
            used = set()
            for s in t:
                match = next(
                    (k for k, q in enumerate(p) if k not in used and abs(q - s) <= tolerance), None
                )
                if match is None:
                    m["onset_fn"] += 1
                else:
                    used.add(match)
                    m["onset_tp"] += 1
            m["onset_fp"] += len(p) - len(used)
    report = {}
    for game, m in games.items():
        report[game] = {
            "examples": int(m["examples"]),
            "button_f1": _f1(m["tp"], m["fp"], m["fn"]),
            "exact_accuracy": m["exact"] / m["examples"],
            "idle_false_positive_rate": m["idle_fp"] / max(1, m["idle"]),
            "onset_precision": m["onset_tp"] / max(1, m["onset_tp"] + m["onset_fp"]),
            "onset_recall": m["onset_tp"] / max(1, m["onset_tp"] + m["onset_fn"]),
            "onset_f1": _f1(m["onset_tp"], m["onset_fp"], m["onset_fn"]),
            "true_onsets": int(m["onset_tp"] + m["onset_fn"]),
            "mouse_mae_px": m["mouse_error"] / m["moving"] if m["moving"] else None,
        }
    keys = ["button_f1", "exact_accuracy", "idle_false_positive_rate", "onset_f1"]
    report["macro"] = {k: float(np.mean([report[g][k] for g in games])) for k in keys}
    mice = [report[g]["mouse_mae_px"] for g in games if report[g]["mouse_mae_px"] is not None]
    report["macro"]["mouse_mae_px"] = float(np.mean(mice)) if mice else None
    return report


def baselines(rows, truth, true_mouse):
    """Repeat-previous-recorded-action and no-input references on the same frames."""
    order = defaultdict(list)
    for i, r in enumerate(rows):
        order[r["sequence"]].append(i)
    repeat, repeat_mouse = list(truth), np.array(true_mouse, dtype=float)
    for idx in order.values():
        idx.sort(key=lambda i: rows[i]["step"])
        repeat[idx[0]] = frozenset()
        repeat_mouse[idx[0]] = 0
        for a, b in zip(idx[:-1], idx[1:], strict=True):
            repeat[b], repeat_mouse[b] = truth[a], true_mouse[a]
    empty = [frozenset()] * len(rows)
    return {
        "repeat_previous": evaluate_actions(rows, repeat, truth, repeat_mouse, true_mouse),
        "no_input": evaluate_actions(rows, empty, truth, np.zeros_like(repeat_mouse), true_mouse),
    }
