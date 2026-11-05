import logging
import sys
from pathlib import Path

import pytest
from conftest import write_jsonl

from voxcanvas import export
from voxcanvas.utils import read_jsonl


def make_row(**overrides: object) -> dict:
    row = {
        "id": "single_span/deletion/EN_B00000_S00000_W000000",
        "task": "single_span",
        "subset": "deletion",
        "edit_type": "deletion",
        "language": "en",
        "uuid": "EN_B00000_S00000_W000000",
        "audio_path": "audio/EN_B00000_S00000_W000000.mp3",
        "align_word": [["hello", 0.0, 0.4], ["big", 0.4, 0.7], ["world", 0.7, 1.2]],
        "original_text": "Hello big world.",
        "edited_text": "Hello world.",
        "instruction": "Delete 'big'.",
        "reverse_instruction": "Add 'big' before 'world'.",
        "edits": [{"op": "deletion", "target": "big", "new": "", "direction": "", "region": [6, 9]}],
        "num_edits": 1,
        "meta": {},
        "status": "ok",
        "synthetic_audio": "audio/single_span/deletion/EN_B00000_S00000_W000000.wav",
        "original_duration": 1.3,
        "synthetic_duration": 1.1,
        "edit_regions": [{"original_start": 0.4, "original_end": 0.7, "synthetic_start": 0.4, "synthetic_end": 0.5}],
        "asr_text": "Hello world.",
        "asr_score": 0.0,
        "sim": 0.91,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({}, None),
        ({"synthetic_duration": 30.0}, None),
        ({"synthetic_duration": 30.01}, "over_max_duration"),
        ({"asr_score": 0.05}, "asr_mismatch"),
        ({"sim": 0.5}, "low_sim"),
        ({"sim": 0.5001}, None),
        ({"synthetic_duration": 31.0, "asr_score": 1.0, "sim": 0.1}, "over_max_duration"),
        ({"asr_score": 1.0, "sim": 0.1}, "asr_mismatch"),
    ],
)
def test_drop_reason_defaults(overrides: dict, reason: str | None) -> None:
    assert export.drop_reason(make_row(**overrides), max_duration=30.0, max_asr_score=0.0, min_sim=0.5) == reason


def test_drop_reason_uses_status_of_failed_synthesis() -> None:
    row = {"status": "too_long", "task": "multi_span", "subset": "multi", "language": "zh"}
    assert export.drop_reason(row, max_duration=30.0, max_asr_score=0.0, min_sim=0.5) == "too_long"


def test_drop_reason_custom_thresholds() -> None:
    row = make_row(asr_score=0.1, sim=0.45, synthetic_duration=35.0)
    assert export.drop_reason(row, max_duration=40.0, max_asr_score=0.2, min_sim=0.4) is None


def test_build_pair_schema() -> None:
    row = make_row()
    pair = export.build_pair(row)
    assert list(pair) == [
        "id",
        "task",
        "language",
        "edit_type",
        "original_audio",
        "synthetic_audio",
        "original_text",
        "synthetic_text",
        "original_duration",
        "synthetic_duration",
        "forward_instruction",
        "reverse_instruction",
        "edit_regions",
        "num_edits",
        "extra",
    ]
    assert list(pair["extra"]) == ["subset", "edits", "meta", "align_word", "asr_text", "asr_score", "sim"]
    assert pair["original_audio"] == row["audio_path"]
    assert pair["synthetic_text"] == row["edited_text"]
    assert pair["forward_instruction"] == row["instruction"]
    assert pair["reverse_instruction"] == row["reverse_instruction"]
    assert pair["extra"]["edits"] == row["edits"] and pair["extra"]["sim"] == row["sim"]


def test_export_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    rows = [
        make_row(id="a"),
        make_row(id="a"),  # merged twice, e.g. after a resume with another --num-shards
        make_row(id="b", asr_score=0.2),
        make_row(id="c", sim=0.3),
        {"id": "d", "task": "single_span", "subset": "deletion", "language": "en", "status": "too_long"},
        make_row(id="e", task="location_reference", subset="d1", language="zh", reverse_instruction=None),
    ]
    write_jsonl(tmp_path / "in.jsonl", rows)
    output = tmp_path / "pairs" / "pairs.jsonl"
    monkeypatch.setattr(sys, "argv", ["export", "--input", str(tmp_path / "in.jsonl"), "--output", str(output)])
    with caplog.at_level(logging.INFO):
        export.main()

    pairs = list(read_jsonl(str(output)))
    assert [pair["id"] for pair in pairs] == ["a", "e"]
    table = caplog.text.splitlines()
    header = next(line for line in table if line.startswith("task"))
    assert header.split() == [
        "task",
        "subset",
        "language",
        "total",
        "kept",
        "asr_mismatch",
        "duplicate",
        "low_sim",
        "too_long",
    ]
    assert next(line for line in table if line.startswith("single_span")).split() == [
        "single_span",
        "deletion",
        "en",
        "5",
        "1",
        "1",
        "1",
        "1",
        "1",
    ]
    assert next(line for line in table if line.startswith("all")).split() == ["all", "6", "2", "1", "1", "1", "1"]
