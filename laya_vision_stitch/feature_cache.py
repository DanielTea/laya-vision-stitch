"""Content-addressed frozen-vision cache with bounded resident feature memory."""

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np


class CachedExamples:
    def __init__(self, runtime, rows, directory, resident=8):
        from .policy_training import fingerprint

        self.entries, self.memory = [], OrderedDict()
        self.resident = resident
        if resident < 1:
            raise ValueError("Cache residency must be positive")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Hash actual weights, not just a claimed model ID in metadata.
        identity = {
            "format": 1,
            "vision": fingerprint(runtime.module.vision),
            "width": runtime.module.policy_config.image_width,
            "processor": runtime.processor.to_dict(),
        }
        for index, original in enumerate(rows):
            row = dict(original)
            if row.get("description"):
                ids = runtime.agent.tok(row["description"], add_special_tokens=False)["input_ids"]
                if len(ids) > runtime.module.policy_config.visual_slots:
                    raise ValueError("Description exceeds visual slots")
                row["_alignment_ids"] = ids
            frame_key = [(f["sha256"], f["age_seconds"]) for f in row["frames"]]
            digest = hashlib.sha256(
                json.dumps([identity, frame_key], sort_keys=True, default=str).encode()
            ).hexdigest()
            path = directory / f"{digest}.npz"
            if not path.exists():
                features, coords = runtime.features(row)
                with tempfile.NamedTemporaryFile(
                    dir=directory, suffix=".tmp", delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    try:
                        np.savez(
                            handle, features=np.asarray(features), coordinates=np.asarray(coords)
                        )
                        handle.flush()
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
            self.entries.append((row, path, runtime.prepare(row)))
            if (index + 1) % 100 == 0:
                print(f"Cached {index + 1}/{len(rows)} examples", flush=True)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row, path, prepared = self.entries[index]
        if path not in self.memory:
            with np.load(path, allow_pickle=False) as data:
                features, coords = data["features"], data["coordinates"]
                if features.ndim != 2 or coords.shape != (len(features), 4):
                    raise ValueError(f"Invalid feature cache shape: {path}")
                if not np.isfinite(features).all() or not np.isfinite(coords).all():
                    raise ValueError(f"Nonfinite feature cache: {path}")
                self.memory[path] = mx.array(features), mx.array(coords)
            if len(self.memory) > self.resident:
                self.memory.popitem(last=False)
        self.memory.move_to_end(path)
        return row, (*self.memory[path], *prepared)
