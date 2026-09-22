"""Persistent localhost screenshot-to-decision API. No OS inputs are sent."""

import argparse
import base64
import binascii
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from .trainable_model import TrainableRuntime

MAX_BODY = 16 * 1024 * 1024


def request_row(payload, config):
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("goal"), str)
        or not payload["goal"].strip()
    ):
        raise ValueError("Provide a nonempty goal")
    if not isinstance(payload.get("controls", ""), str) or not isinstance(
        payload.get("previous_actions", []), list
    ):
        raise ValueError("controls must be text and previous_actions a list")
    frames = payload.get("frames", [])
    if not isinstance(frames, list) or not 1 <= len(frames) <= config.max_frames:
        raise ValueError("Invalid frame count")
    choices = payload.get("choices")
    if choices is not None and (
        not isinstance(choices, dict)
        or not 2 <= len(choices) <= 32
        or any(not isinstance(v, str) or not v.strip() for v in choices.values())
    ):
        raise ValueError("choices must map 2..32 labels to descriptions")
    converted = []
    ages = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Each frame must be an object")
        age = frame.get("age_seconds")
        if not isinstance(age, (float, int)) or not np.isfinite(age) or age < 0:
            raise ValueError("Invalid frame age")
        try:
            raw = base64.b64decode(frame["image_base64"], validate=True)
            with Image.open(io.BytesIO(raw)) as source:
                if max(source.size) > 8192 or source.width * source.height > 16_000_000:
                    raise ValueError("Image exceeds size limit")
                image = source.convert("RGB")
        except (KeyError, TypeError, binascii.Error, OSError) as exc:
            raise ValueError("Each frame needs a valid base64 image") from exc
        converted.append({"image": image, "age_seconds": age})
        ages.append(age)
    if ages != sorted(ages, reverse=True) or len(set(ages)) != len(ages) or ages[-1] != 0:
        raise ValueError("Frames must be oldest first with distinct ages ending at zero")
    # Supervision and arbitrary filesystem paths are intentionally not accepted.
    return {
        "frames": converted,
        "goal": payload["goal"],
        "controls": payload.get("controls", ""),
        "previous_actions": payload.get("previous_actions", []),
        **({"choices": choices} if choices else {}),
    }


def make_server(runtime, port=8767):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                self.respond(404, {"error": "Unknown endpoint"})
                return
            self.respond(
                200,
                {
                    "ready": True,
                    "training_steps": runtime.metadata["training_steps"],
                    "parameters": runtime.parameter_counts(),
                    "input_events_sent": 0,
                },
            )

        def do_POST(self):
            if self.path != "/predict":
                self.respond(404, {"error": "Unknown endpoint"})
                return
            if self.headers.get_content_type() != "application/json":
                self.respond(415, {"error": "Use application/json"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(size))
                row = request_row(payload, runtime.module.policy_config)
                self.respond(200, runtime.predict(row))
            except (ValueError, UnicodeError) as exc:
                self.respond(400, {"error": str(exc)})
            except Exception as exc:
                self.respond(500, {"error": f"Inference failed: {type(exc).__name__}"})

    # Single inference worker: avoids concurrent mutable MLX execution/VRAM spikes.
    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    runtime = TrainableRuntime.load(args.bundle)
    server = make_server(runtime, args.port)
    print(f"Model loaded. http://127.0.0.1:{server.server_port}/predict", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
