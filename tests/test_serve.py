import base64
import io
import json
import threading
import urllib.request
from types import SimpleNamespace

import pytest
from PIL import Image

pytest.importorskip("mlx.core")

from laya_vision_stitch.serve import make_server, request_row  # noqa: E402
from laya_vision_stitch.trainable_model import PolicyConfig  # noqa: E402


def payload():
    data = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(data, format="PNG")
    return {
        "goal": "Move toward red",
        "frames": [{"image_base64": base64.b64encode(data.getvalue()).decode(), "age_seconds": 0}],
        "choices": {"left": "Move left", "right": "Move right"},
        "description": "must not enter inputs",
    }


def test_image_request_is_in_memory_and_excludes_supervision():
    row = request_row(payload(), PolicyConfig())
    assert row["frames"][0]["image"].size == (16, 16)
    assert "description" not in row
    bad = payload()
    bad["frames"][0]["image_base64"] = "not an image"
    with pytest.raises(ValueError):
        request_row(bad, PolicyConfig())
    bad = payload()
    bad["frames"][0]["age_seconds"] = 1
    with pytest.raises(ValueError, match="ending at zero"):
        request_row(bad, PolicyConfig())


def test_persistent_api_reuses_runtime_for_multiple_requests():
    calls = []

    def predict(row):
        calls.append(row["goal"])
        return {"input_events_sent": 0, "goal": row["goal"]}

    runtime = SimpleNamespace(
        metadata={"training_steps": 1},
        module=SimpleNamespace(policy_config=PolicyConfig()),
        parameter_counts=lambda: {"trainable": 1},
        predict=predict,
    )
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(root + "/health") as response:
            assert json.load(response)["ready"]
        for goal in ("red", "blue"):
            request = payload()
            request["goal"] = goal
            req = urllib.request.Request(
                root + "/predict",
                data=json.dumps(request).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as response:
                assert json.load(response)["goal"] == goal
        assert calls == ["red", "blue"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
