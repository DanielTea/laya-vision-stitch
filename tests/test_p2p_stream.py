import io
import tarfile

import numpy as np
import pytest

from laya_vision_stitch.p2p_data import candidates, frame_timing
from laya_vision_stitch.p2p_shard import member_identity
from laya_vision_stitch.p2p_stream import LimitedReader
from laya_vision_stitch.vendor.p2p.video_annotation_pb2 import VideoAnnotation


def test_short_initial_mux_frame_is_not_a_missing_frame():
    result = frame_timing([0, 0.0305, 0.0805, 0.1305, 0.1805])
    assert result["irregular_intervals"] == []
    np.testing.assert_allclose(np.diff(result["timestamps"])[1:], 0.05)


def test_internal_jitter_is_explicit_and_nonmonotonic_pts_fail():
    assert frame_timing([0, 0.05, 0.12, 0.15, 0.2])["irregular_intervals"] == [2, 3]
    assert frame_timing([0, 0.05, 0.15])["irregular_intervals"] == [2]
    for stamps in ([0, 0.05, 0.05], [0, 0.05, 0.04], [0, float("nan")]):
        with pytest.raises(ValueError, match="timestamps"):
            frame_timing(stamps)


def test_unknown_goals_are_identified_without_inventing_intent():
    a = VideoAnnotation()
    a.metadata.frames_per_second = 20
    for _ in range(20):
        f = a.frame_annotations.add()
        f.user_action.is_known = True
        f.user_action.mouse.mouse_delta_px.x = 0
    assert candidates(a)[0] == []
    rows, _ = candidates(a, include_unlabelled=True)
    assert rows and all(r["goal"] == "Continue the current activity." for r in rows)
    assert all(r["instruction_provenance"]["annotator"] is None for r in rows)
    a.frame_annotations[5].user_action.mouse.ClearField("mouse_delta_px")
    assert candidates(a, include_unlabelled=True)[1]["missing_raw_mouse"] == 1


def test_tar_members_cannot_escape_or_follow_links():
    identifier = "019a4b10-11b0-76b3-8112-bd6f0a3e3dff"
    safe = tarfile.TarInfo(f"dataset/{identifier}/video.mp4")
    assert member_identity(safe) == (identifier, "video.mp4")
    for name in (
        f"../{identifier}/video.mp4",
        f"/{identifier}/video.mp4",
        "dataset/not-a-uuid/video.mp4",
        f"dataset/{identifier}/payload.py",
    ):
        assert member_identity(tarfile.TarInfo(name)) is None
    safe.type = tarfile.SYMTYPE
    assert member_identity(safe) is None


def test_stream_byte_budget_is_enforced_and_audited():
    import hashlib

    reader = LimitedReader(io.BytesIO(b"abcdefghij"), 5)
    assert reader.read(3) == b"abc"
    assert reader.read(100) == b"de"
    assert reader.count == 5
    assert reader.digest.hexdigest() == hashlib.sha256(b"abcde").hexdigest()
    with pytest.raises(EOFError, match="byte limit"):
        reader.read(1)


def test_combine_verifies_members_and_refuses_duplicates(tmp_path):
    import json

    from laya_vision_stitch.p2p_data import sha256
    from laya_vision_stitch.p2p_shard import REVISION
    from laya_vision_stitch.p2p_stream import combine_sources

    source = tmp_path / "source"
    identifier = "019a4b10-11b0-76b3-8112-bd6f0a3e3dff"
    recording = source / "dataset" / identifier
    recording.mkdir(parents=True)
    (source / "README.md").write_text("Exact source license card")
    row = {"recording": identifier}
    for name, key in (("annotation.proto", "annotation_sha256"), ("video.mp4", "video_sha256")):
        path = recording / name
        path.write_bytes(name.encode())
        row[key] = sha256(path)
    (source / "inventory.json").write_text(
        json.dumps(
            {
                "repository": "elefantai/p2p-full-data",
                "revision": REVISION,
                "recordings": [row],
                "transport": [],
            }
        )
    )
    combine_sources([source], tmp_path / "combined")
    assert (tmp_path / "combined" / "README.md").read_text() == "Exact source license card"
    with pytest.raises(ValueError, match="duplicate recording"):
        combine_sources([source, source], tmp_path / "duplicate")
    (recording / "video.mp4").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        combine_sources([source], tmp_path / "corrupted")


def test_import_excludes_jitter_across_observation_and_target_span(tmp_path, monkeypatch):
    import json

    from laya_vision_stitch import p2p_shard
    from laya_vision_stitch.p2p_data import sha256

    identifier = "019a4b10-11b0-76b3-8112-bd6f0a3e3dff"
    source = tmp_path / "source"
    recording = source / "dataset" / identifier
    recording.mkdir(parents=True)
    a = VideoAnnotation()
    a.metadata.frames_per_second = 20
    a.metadata.env.env = "test-game"
    for _ in range(32):
        f = a.frame_annotations.add()
        f.user_action.is_known = True
        f.user_action.mouse.mouse_delta_px.x = 0
    (recording / "annotation.proto").write_bytes(a.SerializeToString())
    (recording / "video.mp4").write_bytes(b"video")
    (source / "README.md").write_text("Source license")
    (source / "inventory.json").write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "recording": identifier,
                        "group": "recording-group",
                        "annotation_sha256": sha256(recording / "annotation.proto"),
                        "video_sha256": sha256(recording / "video.mp4"),
                    }
                ]
            }
        )
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"recordings": [{"recording": identifier, "split": "train", "count": 3}]})
    )

    def extract(video, indices, destination, count, timing):
        destination.mkdir(parents=True)
        for index in indices:
            (destination / f"{index:07d}.png").write_bytes(str(index).encode())
        timing.update(timestamps=[i / 20 for i in range(count)], irregular_intervals=[12, 13])
        return count

    monkeypatch.setattr(p2p_shard, "extract_frames", extract)
    output = tmp_path / "output"
    p2p_shard.build(tmp_path / "unused.tar.gz", source, plan, output, local_source=True)
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert [r["frame_index"] for r in rows] == [4, 24]
    audit = json.loads((output / "audit.json").read_text())
    assert audit["recordings"][0]["excluded"]["irregular_video_interval"] == 1
    assert rows[0]["source"]["action_annotation_index"] == 5
