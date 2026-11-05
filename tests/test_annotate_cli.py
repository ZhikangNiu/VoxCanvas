import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest
from conftest import write_jsonl
from openai.resources.chat.completions import Completions

from voxcanvas.annotate import __main__ as cli
from voxcanvas.annotate.llm import LLMClient, draw_answer
from voxcanvas.annotate.records import SUBSETS_OF_TASK, build_record
from voxcanvas.annotate.templates import DELETION
from voxcanvas.text import apply_edits, tokenize_content
from voxcanvas.utils import rng_for


def make_candidate(row: dict, subset: str, rng: random.Random) -> dict:
    return {"row": row, "length_bucket": "4-7", "draw": rng.random()}


def complete_candidate(candidate: dict, llm: LLMClient | None, rng: random.Random) -> dict | None:
    row = candidate["row"]
    word, start, end = tokenize_content(row["text"], row["language"])[-1]
    edits = [{"op": "deletion", "target": word, "new": "", "direction": "", "region": [start, end]}]
    edited = apply_edits(row["text"], edits, row["language"])
    instruction = DELETION[row["language"]].format(x=word)
    meta = {"draws": [candidate["draw"], rng.random()]}
    return build_record(row, "single_span", "deletion", "deletion", edited, instruction, "reverse", edits, meta)


def make_rows(num_rows: int) -> list[dict]:
    align = [[token, 0.0, 0.1] for token in ["we", "went", "to", "the", "old", "market", "today"]]
    text = "We went to the old market today."
    return [
        {"uuid": f"u{i}", "audio_path": "a.mp3", "language": "en", "text": text, "align_word": align}
        for i in range(num_rows)
    ]


def test_cli_writes_in_input_order_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(
        SUBSETS=("deletion",),
        NEEDS_LLM=frozenset(),
        make_candidate=make_candidate,
        complete_candidate=complete_candidate,
    )
    monkeypatch.setattr(cli, "GENERATORS", (fake,))
    rows = [
        {
            "uuid": f"u{i}",
            "audio_path": "a.mp3",
            "language": "en",
            "text": text,
            "align_word": [[token, 0.0, 0.1] for token in align],
        }
        for i, (text, align) in enumerate(
            [
                ("We went to the old market today.", ["we", "went", "to", "the", "old", "market", "today"]),
                ("This row has a broken alignment.", ["this"]),
                ("Then we bought some fresh bread.", ["then", "we", "bought", "some", "fresh", "bread"]),
            ]
        )
    ]
    source, output = tmp_path / "aligned.jsonl", tmp_path / "edits.jsonl"
    write_jsonl(source, rows)
    argv = [
        "annotate",
        "--task",
        "single_span",
        "--subset",
        "deletion",
        "--input",
        str(source),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    cli.main()
    lines = output.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [record["id"] for record in records] == ["single_span/deletion/u0", "single_span/deletion/u2"]
    assert records[0]["edited_text"] == "We went to the old market."
    # Selection and completion draw from separate streams of the row.
    for record in records:
        streams = [rng_for(0, record["uuid"], "deletion", purpose) for purpose in ("candidate", "complete")]
        assert record["meta"]["draws"] == [stream.random() for stream in streams]

    # A run killed mid-write leaves a partial last line; resuming cuts it and redoes that row.
    output.write_text(lines[0] + "\n" + lines[1][:40], encoding="utf-8")
    cli.main()
    assert output.read_text(encoding="utf-8").splitlines() == lines


def test_cli_worker_error_cancels_queued_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started = []

    def failing_complete(candidate: dict, llm: LLMClient | None, rng: random.Random) -> dict | None:
        started.append(candidate["row"]["uuid"])
        time.sleep(0.05)
        raise RuntimeError("endpoint rejected the key")

    fake = SimpleNamespace(
        SUBSETS=("deletion",),
        NEEDS_LLM=frozenset(),
        make_candidate=make_candidate,
        complete_candidate=failing_complete,
    )
    monkeypatch.setattr(cli, "GENERATORS", (fake,))
    source = tmp_path / "aligned.jsonl"
    write_jsonl(source, make_rows(200))
    argv = ["annotate", "--task", "single_span", "--subset", "deletion", "--input", str(source)]
    monkeypatch.setattr(sys, "argv", [*argv, "--output", str(tmp_path / "edits.jsonl"), "--concurrency", "2"])
    with pytest.raises(RuntimeError, match="rejected"):
        cli.main()
    # Without cancel_futures all 200 run; the bound leaves the main thread ~0.5 s to reach the shutdown.
    assert len(started) <= 20, f"{len(started)} of 200 candidates ran after the first failure"


def test_cli_fails_when_every_llm_request_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def complete_with_llm(candidate: dict, llm: LLMClient | None, rng: random.Random) -> dict | None:
        if draw_answer(llm, "prompt", "en", lambda answer: True) is None:
            return None
        return complete_candidate(candidate, llm, rng)

    def reject(self: Completions, **kwargs: object) -> None:
        response = SimpleNamespace(request=None, status_code=400, headers={})
        raise openai.BadRequestError("unknown field top_k", response=response, body=None)

    def reply(self: Completions, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    fake = SimpleNamespace(
        SUBSETS=("deletion",),
        NEEDS_LLM=frozenset({"deletion"}),
        make_candidate=make_candidate,
        complete_candidate=complete_with_llm,
    )
    monkeypatch.setattr(cli, "GENERATORS", (fake,))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    source, output = tmp_path / "aligned.jsonl", tmp_path / "edits.jsonl"
    write_jsonl(source, make_rows(3))
    argv = ["annotate", "--task", "single_span", "--subset", "deletion", "--input", str(source)]
    monkeypatch.setattr(sys, "argv", [*argv, "--output", str(output)])
    monkeypatch.setattr(Completions, "create", reject)
    with pytest.raises(RuntimeError, match="every LLM request failed"):
        cli.main()
    assert output.read_text(encoding="utf-8") == ""
    # Once the endpoint answers, re-running retries the dropped candidates.
    monkeypatch.setattr(Completions, "create", reply)
    cli.main()
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3


def test_cli_rejects_duplicate_uuids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "aligned.jsonl"
    write_jsonl(source, make_rows(2) + make_rows(1))
    argv = ["annotate", "--task", "single_span", "--subset", "deletion", "--input", str(source)]
    monkeypatch.setattr(sys, "argv", [*argv, "--output", str(tmp_path / "edits.jsonl")])
    with pytest.raises(ValueError, match="Duplicate uuid 'u0'"):
        cli.main()


def test_generators_cover_every_subset_exactly_once() -> None:
    generated = sorted(subset for module in cli.GENERATORS for subset in module.SUBSETS)
    assert generated == sorted(subset for subsets in SUBSETS_OF_TASK.values() for subset in subsets)
