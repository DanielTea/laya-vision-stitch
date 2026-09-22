"""Read bounded archive prefixes, keeping only complete recordings with mouse labels."""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, hf_hub_url

from .p2p_data import read_annotation, sha256
from .p2p_shard import REVISION, member_identity


def combine_sources(sources, output):
    """Combine complete pinned recordings without duplicating their video bytes."""
    output.mkdir(parents=True, exist_ok=False)
    records, transport, seen = [], [], set()
    card = None
    for source in sources:
        inventory = json.loads((source / "inventory.json").read_text())
        if (
            inventory["revision"] != REVISION
            or inventory["repository"] != "elefantai/p2p-full-data"
        ):
            raise ValueError("Source revision or repository mismatch")
        source_card = (source / "README.md").read_bytes()
        if card is not None and card != source_card:
            raise ValueError("Source cards differ")
        card = source_card
        for record in inventory["recordings"]:
            identifier = record["recording"]
            # Reuse archive path validation even when reading a local inventory.
            member = tarfile.TarInfo(f"{identifier}/annotation.proto")
            if member_identity(member) != (identifier, "annotation.proto") or identifier in seen:
                raise ValueError("Invalid or duplicate recording")
            seen.add(identifier)
            destination = output / "dataset" / identifier
            destination.mkdir(parents=True)
            for name, key in (
                ("annotation.proto", "annotation_sha256"),
                ("video.mp4", "video_sha256"),
            ):
                path = source / "dataset" / identifier / name
                if sha256(path) != record[key]:
                    raise ValueError("Source member hash mismatch")
                try:
                    os.link(path, destination / name)
                except OSError:
                    shutil.copyfile(path, destination / name)
            records.append(record)
        transport.extend(inventory["transport"])
    if not records or card is None:
        raise ValueError("No complete recordings to combine")
    (output / "README.md").write_bytes(card)
    (output / "inventory.json").write_text(
        json.dumps(
            {
                "repository": "elefantai/p2p-full-data",
                "revision": REVISION,
                "archive": "streamed-prefixes",
                "transport": transport,
                "recordings": records,
            },
            indent=2,
        )
        + "\n"
    )


class LimitedReader:
    def __init__(self, raw, limit):
        self.raw, self.limit, self.count = raw, limit, 0
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        remaining = self.limit - self.count
        if remaining <= 0:
            raise EOFError("Archive prefix byte limit reached")
        data = self.raw.read(min(size if size >= 0 else remaining, remaining))
        self.count += len(data)
        self.digest.update(data)
        return data


def stream_archive(number, count, limit, output):
    archive = f"dataset/batch_{number:05d}.tar.gz"
    url = hf_hub_url("elefantai/p2p-full-data", archive, repo_type="dataset", revision=REVISION)
    records, pending = [], {}
    with requests.get(
        url, headers={"Range": f"bytes=0-{limit - 1}"}, stream=True, timeout=60
    ) as response:
        response.raise_for_status()
        reader = LimitedReader(response.raw, limit)
        try:
            with tarfile.open(fileobj=reader, mode="r|gz") as tar:
                for member in tar:
                    identity = member_identity(member)
                    if identity is None:
                        continue
                    identifier, name = identity
                    if name == "annotation.proto":
                        if member.size > 100_000_000:
                            continue
                        path = output / "dataset" / identifier / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as src:
                            path.write_bytes(src.read())
                        a = read_annotation(path)
                        frames = a.frame_annotations
                        known = sum(f.user_action.is_known for f in frames)
                        raw = sum(
                            f.user_action.is_known
                            and f.user_action.mouse.HasField("mouse_delta_px")
                            for f in frames
                        )
                        system = sum(f.system_action.is_known for f in frames)
                        if (
                            a.metadata.frames_per_second != 20
                            or len(frames) < 800
                            or known < 0.95 * len(frames)
                            or raw < 0.95 * len(frames)
                            or system
                        ):
                            continue
                        group = a.metadata.group_name or a.metadata.id or identifier
                        pending[identifier] = {
                            "recording": identifier,
                            "archive": archive,
                            "game": a.metadata.env.env,
                            "subtype": a.metadata.env.env_subtype,
                            "group": hashlib.sha256(group.encode()).hexdigest(),
                            "actor": hashlib.sha256(a.metadata.user.encode()).hexdigest()
                            if a.metadata.user
                            else None,
                            "timestamp": a.metadata.timestamp,
                            "fps": 20,
                            "frames": len(frames),
                            "human_frames": known,
                            "raw_mouse_frames": raw,
                            "system_frames": system,
                            "text_segments": sum(len(f.frame_text_annotation) for f in frames),
                            "annotation_sha256": sha256(path),
                        }
                    elif identifier in pending:
                        destination = output / "dataset" / identifier / name
                        partial = destination.with_suffix(".partial")
                        try:
                            with tar.extractfile(member) as src, partial.open("wb") as target:
                                while chunk := src.read(1024 * 1024):
                                    target.write(chunk)
                            if partial.stat().st_size != member.size:
                                raise ValueError("Truncated video member")
                            partial.replace(destination)
                        finally:
                            partial.unlink(missing_ok=True)
                        row = pending.pop(identifier)
                        row.update(video_bytes=member.size, video_sha256=sha256(destination))
                        records.append(row)
                        print(
                            f"Collected {number}: {len(records)}/{count} {row['game']}/{row['subtype']}",
                            flush=True,
                        )
                        if len(records) == count:
                            break
        except (EOFError, tarfile.ReadError) as exc:
            print(f"Prefix ended after {len(records)} complete recordings: {exc}", flush=True)
    return records, {
        "archive": archive,
        "prefix_bytes": reader.count,
        "prefix_sha256": reader.digest.hexdigest(),
        "requested_recordings": count,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--archive", type=int)
    mode.add_argument("--combine", type=Path, nargs="+")
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--max-bytes", type=int, default=1_000_000_000)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.combine:
        combine_sources(args.combine, args.output)
        return
    if (
        not 1 <= args.archive <= 545
        or not 1 <= args.count <= 8
        or not 1 <= args.max_bytes <= 2_000_000_000
    ):
        p.error("Invalid bounded download parameters")
    args.output.mkdir(parents=True, exist_ok=False)
    hf_hub_download(
        "elefantai/p2p-full-data",
        "README.md",
        repo_type="dataset",
        revision=REVISION,
        local_dir=args.output,
    )
    records, transport = stream_archive(args.archive, args.count, args.max_bytes, args.output)
    (args.output / "inventory.json").write_text(
        json.dumps(
            {
                "repository": "elefantai/p2p-full-data",
                "revision": REVISION,
                "archive": "streamed-prefixes",
                "transport": [transport],
                "recordings": records,
            },
            indent=2,
        )
        + "\n"
    )
    if len(records) < args.count:
        raise RuntimeError("Incomplete requested sample; complete members and audit were retained")


if __name__ == "__main__":
    main()
