import json
import random
from collections.abc import Callable
from pathlib import Path

from voxcanvas.text import gate, tokenize_aligned


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


class ScriptedLLM:
    """Stands in for LLMClient: returns (or raises) the scripted replies in order and records the requests."""

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def complete(self, prompt: str, system: str) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_row(text: str, lang: str, times: list[tuple[float, float]] | None = None) -> dict:
    """Aligned row that passes the gate; by default every align token lasts 0.1 s followed by a 0.1 s pause."""
    tokens = [token for token, _, _ in tokenize_aligned(text, lang)]
    if times is None:
        times = [(i * 0.2, i * 0.2 + 0.1) for i in range(len(tokens))]
    align = [[token, start, end] for token, (start, end) in zip(tokens, times, strict=True)]
    row = {"uuid": "u1", "audio_path": "a.mp3", "language": lang, "text": text, "align_word": align}
    assert gate(row)
    return row


def candidate_with(
    make_candidate: Callable[[dict, str, random.Random], dict | None], row: dict, subset: str, **wanted: object
) -> dict:
    """First candidate over seeds 0..999 whose fields (or its edit's, e.g. direction) equal `wanted`."""
    for seed in range(1000):
        candidate = make_candidate(row, subset, random.Random(seed))
        if candidate is None:
            continue
        fields = {**candidate["edit"], **candidate}
        if all(fields[key] == value for key, value in wanted.items()):
            return candidate
    raise AssertionError(f"no {subset} candidate with {wanted}")
