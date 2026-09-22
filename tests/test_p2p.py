import json
from pathlib import Path

import numpy as np
import pytest

from laya_vision_stitch.p2p_data import (
    ANNOTATOR,
    candidates,
    extract_frames,
    human_action,
    instructions,
)
from laya_vision_stitch.vendor.p2p.video_annotation_pb2 import FrameAnnotation, VideoAnnotation


def frame(keys=(), dx=0, dy=0):
    f = FrameAnnotation()
    f.user_action.is_known = True
    f.user_action.keyboard.keys.extend(keys)
    f.user_action.mouse.mouse_delta_px.x = dx
    f.user_action.mouse.mouse_delta_px.y = dy
    return f


def annotation():
    a = VideoAnnotation()
    a.metadata.frames_per_second = 20
    a.frame_annotations.extend([frame() for _ in range(20)])
    text = a.frame_annotations[0].frame_text_annotation.add()
    text.instruction = "Walk towards the door."
    text.duration = 1
    text.frame_text_annotator.version = ANNOTATOR
    return a


def test_causal_action_alignment_and_history():
    a = annotation()
    a.frame_annotations[4].CopyFrom(frame(["w"], 10))
    a.frame_annotations[5].CopyFrom(frame(["s"], -20))
    rows, _ = candidates(a)
    r = rows[0]
    assert r["frame_index"] == 4
    assert r["action"]["buttons"] == ["s"]
    assert r["previous_actions"][0]["buttons"] == ["w"]
    assert r["action"]["mouse_delta"] == [-20 / 512, 0]


def test_unknown_is_not_idle_and_conflicts_are_rejected():
    f = frame()
    assert human_action(f)["buttons"] == []
    f.user_action.is_known = False
    with pytest.raises(ValueError, match="unknown_or_system"):
        human_action(f)
    f.user_action.is_known = True
    f.system_action.is_known = True
    with pytest.raises(ValueError, match="unknown_or_system"):
        human_action(f)


def test_absolute_cursor_never_becomes_camera_motion():
    f = frame()
    f.user_action.mouse.mouse_absolute_px.x = 900
    assert human_action(f)["mouse_delta"] == [0, 0]
    assert human_action(f)["pointer_xy"] is None
    f.user_action.mouse.ClearField("mouse_delta_px")
    with pytest.raises(ValueError, match="missing_raw_mouse"):
        human_action(f)


def test_mapping_releases_and_unsupported_controls():
    f = frame(["Space", "LeftShift", "w"])
    f.user_action.mouse.buttons_down.append("1")
    assert human_action(f)["buttons"] == ["mouse_right", "shift", "space", "w"]
    assert human_action(frame())["buttons"] == []
    f.user_action.mouse.scroll_delta_px.y = 1
    with pytest.raises(ValueError, match="scroll"):
        human_action(f)
    with pytest.raises(ValueError, match="unknown_keyboard"):
        human_action(frame(["F12"]))
    with pytest.raises(ValueError, match="mouse_out_of_range"):
        human_action(frame(dx=513))


def test_text_does_not_leak_backwards_and_overlaps_are_excluded():
    a = annotation()
    a.frame_annotations[0].ClearField("frame_text_annotation")
    text = a.frame_annotations[5].frame_text_annotation.add()
    text.instruction = "Turn left."
    text.duration = 0.25
    text.frame_text_annotator.version = ANNOTATOR
    goals = instructions(a)
    assert goals[:5] == [None] * 5
    assert goals[5]["text"] == "Turn left."
    assert goals[10] is None
    other = a.frame_annotations[8].frame_text_annotation.add()
    other.CopyFrom(text)
    other.instruction = "Turn right."
    goals = instructions(a)
    assert goals[8:10] == [False, False]
    assert goals[10]["text"] == "Turn right."
    rows, _ = candidates(a, stride=1)
    assert not any(r["frame_index"] in (4, 7, 8, 9) for r in rows)


def test_video_count_mismatch_is_not_silent(tmp_path):
    av = pytest.importorskip("av")
    video = tmp_path / "video.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for _ in range(6):
            image = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")
            for packet in stream.encode(image):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    assert extract_frames(video, [4, 5], tmp_path / "frames", 6) == 6
    with pytest.raises(ValueError, match="count mismatch"):
        extract_frames(video, [4, 5], tmp_path / "bad", 7)


def test_real_import_whole_game_separation():
    path = Path("artifacts/p2p-pilot-data-002")
    if not (path / "audit.json").exists():
        pytest.skip("Local pinned sample not imported")
    from laya_vision_stitch.d2e_data import BUTTONS
    from laya_vision_stitch.policy_data import check_separation, read_manifest
    from laya_vision_stitch.trainable_model import PolicyConfig

    config = PolicyConfig(buttons=tuple(BUTTONS), max_frames=4)
    train, validation = [
        read_manifest(path / f"{s}.jsonl", config) for s in ("train", "validation")
    ]
    check_separation(train, validation, holdout_games=True)
    assert {r["game"] for r in validation} == {"roblox"}
    assert json.loads((path / "audit.json").read_text())["revision"].startswith("305ebc8")


def test_staged_training_updates_camera_without_updating_frozen_backbone():
    import io
    from types import SimpleNamespace

    import mlx.core as mx
    import mlx.nn as nn

    from laya_vision_stitch.p2p_training import train_stage
    from laya_vision_stitch.policy_training import fingerprint
    from laya_vision_stitch.trainable_model import ActionHeads, PolicyConfig

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_config = PolicyConfig(
                buttons=("w", "a"), action_chunk_size=2, connector_width=8, heads=2
            )
            self.vision = nn.Linear(8, 8)
            self.connector = nn.Linear(8, 8)
            self.actions = ActionHeads(8, self.policy_config)
            self.laya = SimpleNamespace(encoder=SimpleNamespace(layers=[]))

        def action_context(self, x):
            return self.connector(x), mx.array([0])

        def from_features(self, x):
            h, _ = self.action_context(x)
            return self.actions(h[:, 0], h)

    mx.random.seed(41)
    model = TinyModel()
    runtime = SimpleNamespace(
        module=model, metadata={"training_steps": 0, "action_training_examples": 0}
    )
    rows = [
        {"action": {"buttons": [key], "mouse_delta": [motion, 0]}}
        for key, motion in (("w", 0.1), ("a", -0.1))
    ]
    examples = [(r, (mx.random.normal((1, 2, 8)),)) for r in rows]
    frozen = fingerprint(model.vision)
    camera = fingerprint(model.actions.chunk_mouse)
    connector = fingerprint(model.connector)
    for adapters in (False, True):
        train_stage(runtime, rows, examples, 2, 1e-3, 41, adapters, io.StringIO())
        assert fingerprint(model.vision) == frozen
        if not adapters:
            assert fingerprint(model.connector) == connector
    assert fingerprint(model.actions.chunk_mouse) != camera
    assert fingerprint(model.connector) != connector
    assert runtime.metadata["action_training_examples"] == 16
    button_logits = model.from_features(*examples[0][1])["buttons"]
    mx.eval(button_logits)
    camera_before = fingerprint(model.actions.chunk_mouse)
    upstream_before = fingerprint(model.connector)
    train_stage(runtime, rows, examples, 2, 1e-3, 41, False, io.StringIO(), camera_only=True)
    np.testing.assert_array_equal(
        np.asarray(button_logits), np.asarray(model.from_features(*examples[0][1])["buttons"])
    )
    assert fingerprint(model.connector) == upstream_before
    assert fingerprint(model.actions.chunk_mouse) != camera_before
