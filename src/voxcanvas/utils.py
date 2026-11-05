import json
import logging
import os
import random
import zlib
from collections.abc import Iterable, Iterator


logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def check_uuids(rows: Iterable[dict], path: str) -> None:
    """Raise ValueError unless every uuid in `rows` (read from `path`) is a non-empty string without "/" seen once.

    uuids are part of every record id and wav path, and align's resume key.
    """
    seen = set()
    for row in rows:
        uuid = row["uuid"]
        if not isinstance(uuid, str) or not uuid or "/" in uuid:
            raise ValueError(f"Invalid uuid {uuid!r} in {path}: it must be a non-empty string without '/'")
        if uuid in seen:
            raise ValueError(f"Duplicate uuid {uuid!r} in {path}")
        seen.add(uuid)


def load_done_ids(path: str, key: str = "id") -> set[str]:
    """Ids already written to an output jsonl, used to resume interrupted runs.

    A partial last line (a run killed mid-write) is cut off so that the next append starts on a fresh line.
    """
    if not os.path.exists(path):
        return set()
    done, num_complete_bytes = set(), 0
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            num_complete_bytes += len(line)
            if line.strip():
                done.add(json.loads(line)[key])
    num_partial_bytes = os.path.getsize(path) - num_complete_bytes
    if num_partial_bytes:
        logger.warning(f"Cutting a partial last line ({num_partial_bytes} bytes) from {path}")
        os.truncate(path, num_complete_bytes)
    return done


def read_shard(
    input_path: str, output_path: str, shard_id: int, num_shards: int, key: str = "id"
) -> tuple[list[dict], int]:
    """Rows of this shard that are not in `output_path` yet, and how many ids are already there.

    A row belongs to shard `zlib.crc32(row[key]) % num_shards`, so it keeps its shard (and resumes) when the input
    grows or is reordered. The output file (and its directory) is created, so a shard without rows still leaves one.
    """
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"--shard-id must be in [0, {num_shards}), got {shard_id}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    open(output_path, "a").close()
    done = load_done_ids(output_path, key)
    rows = [
        row
        for row in read_jsonl(input_path)
        if zlib.crc32(row[key].encode()) % num_shards == shard_id and row[key] not in done
    ]
    return rows, len(done)


def check_checkpoints(*paths: str) -> None:
    """Raise FileNotFoundError for a missing path under checkpoints/; other values (hub ids, URLs) are not checked."""
    for path in paths:
        if path.startswith("checkpoints/") and not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found; download the checkpoints with `bash scripts/download_checkpoints.sh` "
                "and run from the repository root"
            )


def rng_for(seed: int, *keys: str) -> random.Random:
    """Independent, process-stable random stream per (seed, keys)."""
    return random.Random("-".join([str(seed), *keys]))
