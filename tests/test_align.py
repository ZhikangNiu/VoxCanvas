import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from conftest import write_jsonl
from qwen_asr.inference.qwen3_asr import ASRTranscription
from qwen_asr.inference.qwen3_forced_aligner import ForcedAlignItem, ForcedAlignResult
from qwen_asr.inference.utils import SUPPORTED_LANGUAGES

from voxcanvas import align
from voxcanvas.align import LANGUAGES, to_aligned_row, transcribe_clips
from voxcanvas.utils import read_jsonl


class FakeModel:
    """Stands in for Qwen3ASRModel: OOMs on any multi-clip batch, cannot read paths containing 'bad'."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> "FakeModel":
        return cls()

    def transcribe(self, audio: list[str], language: list[str], return_time_stamps: bool) -> list[ASRTranscription]:
        assert return_time_stamps
        self.calls.append(audio)
        if len(audio) > 1:
            raise torch.OutOfMemoryError("CUDA out of memory")
        if "bad" in audio[0]:
            raise FileNotFoundError(audio[0])
        items = [ForcedAlignItem(text="hi", start_time=0.0, end_time=0.24)]
        return [ASRTranscription(language=language[0], text=f" {audio[0]} ", time_stamps=ForcedAlignResult(items))]


def test_languages_are_qwen_names() -> None:
    assert set(LANGUAGES.values()) <= set(SUPPORTED_LANGUAGES)


def test_to_aligned_row_zh_chars_and_passthrough() -> None:
    row = {"uuid": "ZH_1", "audio_path": "a.mp3", "language": "zh", "speaker": "S0"}
    items = [
        ForcedAlignItem(text="能", start_time=0.0, end_time=0.16),
        ForcedAlignItem(text="源", start_time=0.16, end_time=0.4),
        ForcedAlignItem(text="ETF", start_time=0.48, end_time=0.48),
    ]
    result = ASRTranscription(language="Chinese", text=" 能源，ETF。\n", time_stamps=ForcedAlignResult(items))
    assert to_aligned_row(row, result) == {
        "uuid": "ZH_1",
        "audio_path": "a.mp3",
        "language": "zh",
        "speaker": "S0",
        "text": "能源，ETF。",
        "align_word": [["能", 0.0, 0.16], ["源", 0.16, 0.4], ["ETF", 0.48, 0.48]],
    }


def test_to_aligned_row_overrides_source_text() -> None:
    row = {"uuid": "EN_1", "audio_path": "a.mp3", "language": "en", "text": " you- No."}
    items = [ForcedAlignItem(text="Didn't", start_time=1.84, end_time=2.08)]
    result = ASRTranscription(language="English", text="Didn't.", time_stamps=ForcedAlignResult(items))
    aligned = to_aligned_row(row, result)
    assert aligned["text"] == "Didn't."
    assert aligned["align_word"] == [["Didn't", 1.84, 2.08]]


def test_to_aligned_row_drops_empty_results() -> None:
    row = {"uuid": "EN_1", "audio_path": "a.mp3", "language": "en"}
    assert to_aligned_row(row, ASRTranscription(language="", text="", time_stamps=None)) is None
    assert to_aligned_row(row, ASRTranscription(language="English", text="...", time_stamps=None)) is None
    no_items = ForcedAlignResult([])
    assert to_aligned_row(row, ASRTranscription(language="English", text="hi", time_stamps=no_items)) is None


def test_transcribe_clips_retries_one_by_one_and_skips_failures() -> None:
    model = FakeModel()
    audio, language = ["ok1.mp3", "bad.mp3", "ok2.mp3"], ["English", "Chinese", "Chinese"]
    results = transcribe_clips(model, audio, language, return_time_stamps=True)
    assert model.calls == [["ok1.mp3", "bad.mp3", "ok2.mp3"], ["ok1.mp3"], ["bad.mp3"], ["ok2.mp3"]]
    assert results[1] is None
    assert [r.text for r in (results[0], results[2])] == [" ok1.mp3 ", " ok2.mp3 "]
    assert [r.language for r in (results[0], results[2])] == ["English", "Chinese"]


def test_transcribe_clips_single_batch_call() -> None:
    model = FakeModel()
    results = transcribe_clips(model, ["ok.mp3"], ["Chinese"], return_time_stamps=True)
    assert model.calls == [["ok.mp3"]]
    expected = {"uuid": "a", "text": "ok.mp3", "align_word": [["hi", 0.0, 0.24]]}
    assert to_aligned_row({"uuid": "a"}, results[0]) == expected


def test_cli_shards_by_uuid_hash_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"uuid": f"u{i}", "audio_path": f"ok{i}.mp3", "language": "en"} for i in (0, 4, 1, 5)]
    write_jsonl(tmp_path / "manifest.jsonl", rows)
    output = tmp_path / "out.jsonl"
    write_jsonl(output, [{**rows[0], "text": "done before", "align_word": [["hi", 0.0, 0.2]]}])
    monkeypatch.setattr(align, "Qwen3ASRModel", FakeModel)
    argv = ["align", "--input", str(tmp_path / "manifest.jsonl"), "--output", str(output), "--batch-size", "1"]
    argv += ["--asr-model", "Qwen/Qwen3-ASR-1.7B", "--aligner-model", "Qwen/Qwen3-ForcedAligner-0.6B"]
    monkeypatch.setattr(sys, "argv", [*argv, "--shard-id", "0", "--num-shards", "2"])
    align.main()

    # Shard 0 of 2 holds the uuids with an even crc32: u0 (already done) and u1.
    assert [zlib.crc32(row["uuid"].encode()) % 2 for row in rows] == [0, 1, 0, 1]
    out = [(row["uuid"], row["text"]) for row in read_jsonl(str(output))]
    assert out == [("u0", "done before"), ("u1", "ok1.mp3")]


def test_cli_rejects_duplicate_uuids_before_loading_the_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"uuid": "u0", "audio_path": f"ok{i}.mp3", "language": "en"} for i in range(2)]
    write_jsonl(tmp_path / "manifest.jsonl", rows)
    monkeypatch.setattr(align, "Qwen3ASRModel", None)
    argv = ["align", "--input", str(tmp_path / "manifest.jsonl"), "--output", str(tmp_path / "out.jsonl")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="Duplicate uuid 'u0'"):
        align.main()


def test_cli_loads_the_downloaded_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_jsonl(tmp_path / "manifest.jsonl", [{"uuid": "u0", "audio_path": "ok0.mp3", "language": "en"}])
    loaded = []

    def from_pretrained(path: str, forced_aligner: str, **kwargs: object) -> FakeModel:
        loaded.append((path, forced_aligner))
        return FakeModel()

    monkeypatch.setattr(align, "Qwen3ASRModel", SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setattr(sys, "argv", ["align", "--input", "manifest.jsonl", "--output", "out.jsonl"])
    with pytest.raises(FileNotFoundError, match=r"checkpoints/Qwen3-ASR-1\.7B not found; .*download_checkpoints\.sh"):
        align.main()
    (tmp_path / "checkpoints" / "Qwen3-ASR-1.7B").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=r"checkpoints/Qwen3-ForcedAligner-0\.6B not found"):
        align.main()
    (tmp_path / "checkpoints" / "Qwen3-ForcedAligner-0.6B").mkdir()
    align.main()
    assert loaded == [("checkpoints/Qwen3-ASR-1.7B", "checkpoints/Qwen3-ForcedAligner-0.6B")]
