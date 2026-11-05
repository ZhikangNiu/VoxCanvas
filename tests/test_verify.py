import sys
import zlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from conftest import write_jsonl

from voxcanvas.utils import read_jsonl
from voxcanvas.verify import asr, sim


def test_to_jiwer_input_en_keeps_apostrophes_and_drops_punctuation() -> None:
    assert asr.to_jiwer_input("Well-known, it's 5 O'Clock!  Yes—", "en") == "well known it's 5 o'clock yes"


def test_to_jiwer_input_zh_drops_whitespace_punctuation_and_symbols() -> None:
    assert asr.to_jiwer_input("你好， 世界！“Chat GPT”+1%。", "zh") == "ni3 hao3 shi4 jie4 ChatGPT1"


def test_to_jiwer_input_uses_tone_numbered_pinyin() -> None:
    assert asr.to_jiwer_input("我的书，好。", "zh") == "wo3 de5 shu1 hao3"
    assert asr.to_jiwer_input("Hello, World.", "en") == "hello world"


@pytest.mark.parametrize(
    ("lang", "ref", "hyp", "score"),
    [
        ("en", "However all that is changed by events.", "However, all that is changed by events.", 0.0),
        ("en", "a b c d", "a x c d", 0.25),
        ("en", "five apples", "5 apples", 0.5),
        ("en", "a b", "", 1.0),
        ("zh", "他来了。", "她来了！", 0.0),
        ("zh", "我买书", "我卖书", 1 / 3),
        ("zh", "用ChatGPT写", "用chatgpt写", 1 / 3),
        # Reference scores of real synthesized rows: (edited_text, ASR text, asr_score).
        ("zh", "他既然因重孝期内，请旨赐婚一事，人心。", "他既然因重孝期内停止赐婚一事，人心。", 0.0625),
        ("zh", "到底是谁这么有能耐，把十一讨厌上非常了女人。", "到底是谁这么有能耐，把十一讨厌上非常的女人？", 0.05),
        (
            "zh",
            "你这个家伙名叫马峰，他之前曾是一个诈骗集团的规模相当庞大的成员，前不久才刚刚假释。",
            "你这这家伙名叫马峰，他之前曾是一个诈骗集团的规模相当庞大的成员，前不久才刚刚假释。",
            0.02631578947368421,
        ),
        (
            "zh",
            "股东大会最重要董事，组成董事会。董事会负责东印度公司的日常经营管理。",
            "股东大会最重要，董事组成董事会，董事会负责东印度公司的日常经营管理。",
            0.0,
        ),
    ],
)
def test_compute_asr_score(lang: str, ref: str, hyp: str, score: float) -> None:
    assert asr.compute_asr_score(ref, hyp, lang) == pytest.approx(score)


class FakeASRModel:
    """Stands in for Qwen3ASRModel: fails the whole batch when any wav has no transcript, as an unreadable file does."""

    transcripts = {"a.wav": "Hello, world.", "b.wav": "你好世界", "c.wav": "hello word"}

    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> "FakeASRModel":
        return cls()

    def transcribe(self, audio: list[str], language: list[None], return_time_stamps: bool) -> list[SimpleNamespace]:
        assert audio, "transcribe must not be called with an empty batch"
        assert language == [None] * len(audio) and not return_time_stamps
        unreadable = [path for path in audio if path not in self.transcripts]
        if unreadable:
            raise FileNotFoundError(unreadable[0])
        return [SimpleNamespace(text=self.transcripts[path]) for path in audio]


def test_asr_cli_scores_ok_rows_passes_others_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # After the resume, the first batch of 2 has no ok row, so it must not reach the model.
    rows = [
        {"id": "r0", "status": "ok", "synthetic_audio": "a.wav", "edited_text": "hello world", "language": "en"},
        {"id": "r1", "status": "too_long", "synthetic_audio": "", "edited_text": "x", "language": "en"},
        {"id": "r2", "status": "no_change", "synthetic_audio": "", "edited_text": "x", "language": "zh"},
        {"id": "r3", "status": "ok", "synthetic_audio": "b.wav", "edited_text": "你好，世界。", "language": "zh"},
        {"id": "r4", "status": "ok", "synthetic_audio": "c.wav", "edited_text": "hello world", "language": "en"},
    ]
    write_jsonl(tmp_path / "in.jsonl", rows)
    output = tmp_path / "out.jsonl"
    write_jsonl(output, [{**rows[0], "asr_text": "done before", "asr_score": 0.0}])
    monkeypatch.setattr(asr, "Qwen3ASRModel", FakeASRModel)
    argv = ["asr", "--input", str(tmp_path / "in.jsonl"), "--output", str(output), "--batch-size", "2"]
    monkeypatch.setattr(sys, "argv", [*argv, "--model", "Qwen/Qwen3-ASR-1.7B"])
    asr.main()

    out = {row["id"]: row for row in read_jsonl(str(output))}
    assert list(out) == ["r0", "r1", "r2", "r3", "r4"]
    assert out["r0"]["asr_text"] == "done before"
    assert out["r1"] == rows[1] and out["r2"] == rows[2]
    assert out["r3"]["asr_text"] == "你好世界" and out["r3"]["asr_score"] == 0.0
    assert out["r4"]["asr_score"] == 0.5


def test_asr_cli_writes_every_other_row_and_retries_an_unreadable_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {"id": "r0", "status": "ok", "synthetic_audio": "a.wav", "edited_text": "hello world", "language": "en"},
        {"id": "r1", "status": "ok", "synthetic_audio": "bad.wav", "edited_text": "hello world", "language": "en"},
        {"id": "r2", "status": "no_context", "synthetic_audio": "", "edited_text": "x", "language": "en"},
        {"id": "r3", "status": "ok", "synthetic_audio": "b.wav", "edited_text": "你好，世界。", "language": "zh"},
    ]
    write_jsonl(tmp_path / "in.jsonl", rows)
    output = tmp_path / "out.jsonl"
    monkeypatch.setattr(asr, "Qwen3ASRModel", FakeASRModel)
    argv = ["asr", "--input", str(tmp_path / "in.jsonl"), "--output", str(output), "--batch-size", "2"]
    monkeypatch.setattr(sys, "argv", [*argv, "--model", "Qwen/Qwen3-ASR-1.7B"])
    # The failed batch is retried clip by clip; only r1 is left undone, then the run fails so nothing is merged.
    with pytest.raises(RuntimeError, match="1 synthetic wavs could not be transcribed"):
        asr.main()
    assert [row["id"] for row in read_jsonl(str(output))] == ["r0", "r2", "r3"]

    monkeypatch.setitem(FakeASRModel.transcripts, "bad.wav", "hello world")
    asr.main()
    out = {row["id"]: row for row in read_jsonl(str(output))}
    assert list(out) == ["r0", "r2", "r3", "r1"]
    assert out["r1"]["asr_score"] == 0.0 and out["r0"]["asr_score"] == 0.0


def run_asr_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict], num_shards: int) -> list[str]:
    """Run (or resume) every shard of verify.asr on `rows`; return the ids of the merged shard outputs."""
    write_jsonl(tmp_path / "in.jsonl", rows)
    monkeypatch.setattr(asr, "Qwen3ASRModel", FakeASRModel)
    merged = []
    for shard_id in range(num_shards):
        output = tmp_path / "asr" / f"out.jsonl.shard{shard_id}"
        argv = ["asr", "--input", str(tmp_path / "in.jsonl"), "--output", str(output), "--model", "Qwen/Qwen3-ASR-1.7B"]
        monkeypatch.setattr(sys, "argv", [*argv, "--shard-id", str(shard_id), "--num-shards", str(num_shards)])
        asr.main()
        shard_ids = [row["id"] for row in read_jsonl(str(output))]
        assert all(zlib.crc32(row_id.encode()) % num_shards == shard_id for row_id in shard_ids)
        merged += shard_ids
    return merged


def test_asr_cli_shards_by_id_hash_and_resumes_a_grown_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def make_row(row_id: str) -> dict:
        return {"id": row_id, "status": "ok", "synthetic_audio": "a.wav", "edited_text": "hello", "language": "en"}

    rows = [make_row(f"b/u{i}") for i in range(40)]
    assert sorted(run_asr_shards(tmp_path, monkeypatch, rows, num_shards=4)) == sorted(row["id"] for row in rows)
    # Rows added in front shift every position; each row must keep its shard so the resumed run adds only the new one.
    grown = [make_row("a/u0"), *rows[:20], make_row("a/u1"), *rows[20:]]
    merged = run_asr_shards(tmp_path, monkeypatch, grown, num_shards=4)
    assert Counter(merged) == Counter(row["id"] for row in grown)


def test_asr_cli_leaves_an_empty_output_for_an_empty_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = {"id": "r0", "status": "ok", "synthetic_audio": "a.wav", "edited_text": "hello world", "language": "en"}
    write_jsonl(tmp_path / "in.jsonl", [row])
    empty_shard = 1 - zlib.crc32(b"r0") % 2
    output = tmp_path / "asr" / f"out.jsonl.shard{empty_shard}"
    monkeypatch.setattr(asr, "Qwen3ASRModel", None)
    argv = ["asr", "--input", str(tmp_path / "in.jsonl"), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", [*argv, "--shard-id", str(empty_shard), "--num-shards", "2"])
    asr.main()
    assert output.read_text() == ""


def test_asr_cli_skips_model_when_all_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = {"id": "r0", "status": "ok", "synthetic_audio": "a.wav", "edited_text": "hello world", "language": "en"}
    write_jsonl(tmp_path / "in.jsonl", [row])
    write_jsonl(tmp_path / "out.jsonl", [{**row, "asr_text": "hello world", "asr_score": 0.0}])
    monkeypatch.setattr(asr, "Qwen3ASRModel", None)
    argv = ["asr", "--input", str(tmp_path / "in.jsonl"), "--output", str(tmp_path / "out.jsonl")]
    monkeypatch.setattr(sys, "argv", argv)
    asr.main()
    assert len(list(read_jsonl(str(tmp_path / "out.jsonl")))) == 1


def test_asr_cli_loads_the_downloaded_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    row = {"id": "r0", "status": "ok", "synthetic_audio": "a.wav", "edited_text": "hello world", "language": "en"}
    write_jsonl(tmp_path / "in.jsonl", [row])
    loaded = []

    def from_pretrained(path: str, **kwargs: object) -> FakeASRModel:
        loaded.append(path)
        return FakeASRModel()

    monkeypatch.setattr(asr, "Qwen3ASRModel", SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setattr(sys, "argv", ["asr", "--input", "in.jsonl", "--output", "out.jsonl"])
    with pytest.raises(FileNotFoundError, match=r"checkpoints/Qwen3-ASR-1\.7B not found; .*download_checkpoints\.sh"):
        asr.main()
    (tmp_path / "checkpoints" / "Qwen3-ASR-1.7B").mkdir(parents=True)
    asr.main()
    assert loaded == ["checkpoints/Qwen3-ASR-1.7B"]


def test_load_wav_16k_downmixes_and_resamples(tmp_path: Path) -> None:
    stereo = np.stack([np.full(24000, 0.2), np.full(24000, 0.4)], axis=1).astype(np.float32)
    sf.write(tmp_path / "stereo.wav", stereo, 24000)
    wav = sim.load_wav_16k(str(tmp_path / "stereo.wav"))
    assert wav.shape == (1, 16000)
    assert wav[0, 4000:12000].mean().item() == pytest.approx(0.3, abs=1e-3)


class FakeSpeakerModel(torch.nn.Module):
    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x.mean(dim=1), x.abs().mean(dim=1), x.std(dim=1)], dim=1) * self.scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sim runs on cuda")
def test_sim_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    t = np.arange(16000) / 16000
    sf.write(tmp_path / "orig.wav", 0.5 * np.sin(2 * np.pi * 220 * t), 16000)
    sf.write(tmp_path / "other.wav", 0.1 + 0.1 * np.sign(np.sin(2 * np.pi * 3 * t)), 16000)
    orig, other = str(tmp_path / "orig.wav"), str(tmp_path / "other.wav")
    late = str(tmp_path / "late.wav")
    rows = [
        {"id": "r0", "status": "ok", "audio_path": orig, "synthetic_audio": orig},
        {"id": "r1", "status": "too_long", "audio_path": orig, "synthetic_audio": ""},
        {"id": "r2", "status": "ok", "audio_path": orig, "synthetic_audio": late},
        {"id": "r3", "status": "ok", "audio_path": orig, "synthetic_audio": other},
    ]
    write_jsonl(tmp_path / "in.jsonl", rows)
    torch.save({"model": {"scale": torch.ones(1)}}, tmp_path / "ckpt.pth")
    monkeypatch.setattr(sim, "ECAPA_TDNN_SMALL", FakeSpeakerModel)
    output = tmp_path / "sim" / "out.jsonl"
    argv = ["sim", "--input", str(tmp_path / "in.jsonl"), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", [*argv, "--ckpt", str(tmp_path / "ckpt.pth")])
    # r2's wav does not exist yet: every other row is written, r2 stays undone and the run fails.
    with pytest.raises(RuntimeError, match="1 rows have unreadable audio"):
        sim.main()
    assert [row["id"] for row in read_jsonl(str(output))] == ["r0", "r1", "r3"]

    sf.write(late, 0.5 * np.sin(2 * np.pi * 220 * t), 16000)
    sim.main()
    out = {row["id"]: row for row in read_jsonl(str(output))}
    assert list(out) == ["r0", "r1", "r3", "r2"]
    assert out["r0"]["sim"] == pytest.approx(1.0) and out["r2"]["sim"] == pytest.approx(1.0)
    assert out["r1"] == rows[1]
    assert out["r3"]["sim"] < 0.9


def test_sim_cli_rejects_a_missing_or_incomplete_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_jsonl(tmp_path / "in.jsonl", [{"id": "r0", "status": "too_long", "audio_path": "", "synthetic_audio": ""}])
    monkeypatch.setattr(sim, "ECAPA_TDNN_SMALL", None)
    monkeypatch.setattr(sys, "argv", ["sim", "--input", "in.jsonl", "--output", "out.jsonl"])
    with pytest.raises(
        FileNotFoundError, match=r"checkpoints/wavlm_large_finetune\.pth not found; .*drive\.google\.com"
    ):
        sim.main()
    (tmp_path / "checkpoints").mkdir()
    torch.save({"model": {}}, tmp_path / "checkpoints" / "wavlm_large_finetune.pth")
    monkeypatch.setattr(sim, "ECAPA_TDNN_SMALL", FakeSpeakerModel)
    with pytest.raises(ValueError, match="checkpoints/wavlm_large_finetune.pth is not a WavLM ECAPA-TDNN checkpoint"):
        sim.main()
