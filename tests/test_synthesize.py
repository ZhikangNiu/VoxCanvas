import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
from conftest import write_jsonl
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import convert_char_to_pinyin

from voxcanvas import synthesize
from voxcanvas.synthesize import build_edit_mask, synthesize_record
from voxcanvas.utils import read_jsonl, rng_for


class FakeCFM:
    """CPU stand-in for CFM: real mel features; sampling returns the condition plus the extra frame CFM adds."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.mel_spec = MelSpec()
        self.sample_kwargs: list[dict] = []

    def sample(self, cond: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
        self.sample_kwargs.append(kwargs)
        return F.pad(cond, (0, 0, 0, 1)), None


class FakeVocos:
    """Decodes every mel frame but the extra one to 256 samples of a linear ramp from `low` to `high`."""

    def __init__(self, low: float = 0.5, high: float = 0.5) -> None:
        self.low, self.high = low, high

    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        return torch.linspace(self.low, self.high, (mel.shape[2] - 1) * 256).unsqueeze(0)


SAMPLING = {"seed": 0, "nfe_step": 32, "cfg_strength": 2.0, "sway_sampling_coef": -1.0}


def make_record(tmp_path: Path, record_id: str, edited_text: str) -> dict:
    """A 2 s, 16 kHz quiet sine clip (rms 0.035) of "We go home now."."""
    audio_path = tmp_path / "clip.wav"
    if not audio_path.exists():
        sf.write(audio_path, 0.05 * np.sin(2 * np.pi * 220 * np.arange(32000) / 16000), 16000)
    return {
        "id": record_id,
        "task": "single_span",
        "language": "en",
        "audio_path": str(audio_path),
        "align_word": [["We", 0.1, 0.5], ["go", 0.5, 0.9], ["home", 0.9, 1.5], ["now", 1.5, 1.9]],
        "original_text": "We go home now.",
        "edited_text": edited_text,
    }


def test_synthesize_record_ok(tmp_path: Path) -> None:
    record = make_record(tmp_path, "single_span/deletion/u0", "We go home.")
    model = FakeCFM()
    row = synthesize_record(record, model, FakeVocos(), str(tmp_path), **SAMPLING)

    # 2 s at 24 kHz = 188 frames; [1.5, 1.9] = frames [141, 178) becomes 9 zero frames -> 160 frames.
    path = str(tmp_path / "single_span" / "deletion" / "u0.wav")
    assert row == {
        **record,
        "status": "ok",
        "synthetic_audio": path,
        "original_duration": 2.0,
        "synthetic_duration": round(160 * 256 / 24000, 4),
        "edit_regions": [{"original_start": 1.5, "original_end": 1.9, "synthetic_start": 1.5, "synthetic_end": 1.6}],
    }
    (kwargs,) = model.sample_kwargs
    assert kwargs["duration"] == 160 and kwargs["steps"] == 32 and kwargs["sway_sampling_coef"] == -1.0
    assert kwargs["seed"] == rng_for(0, "single_span/deletion/u0").getrandbits(32)
    assert kwargs["text"] == convert_char_to_pinyin(["We go home."])
    assert torch.equal(kwargs["edit_mask"], build_edit_mask(160, [[1.5, 1.6]]))

    info = sf.info(path)
    assert (info.subtype, info.samplerate, info.frames) == ("PCM_16", 24000, 160 * 256)
    # The quiet input was boosted to rms 0.1 for sampling; the output is scaled back by rms / 0.1.
    wav, _ = sf.read(path)
    assert wav == pytest.approx(0.5 * 0.05 / np.sqrt(2) / 0.1, abs=1e-3)


def test_synthesize_record_scales_overshoot_instead_of_clipping(tmp_path: Path) -> None:
    record = make_record(tmp_path, "single_span/deletion/u0", "We go home.")
    row = synthesize_record(record, FakeCFM(), FakeVocos(-4.0, 2.0), str(tmp_path), **SAMPLING)
    # Undoing the RMS boost leaves a ramp from -1.41 to 0.71; it is scaled to peak 1, not clipped at -1.
    wav, _ = sf.read(row["synthetic_audio"])
    assert wav == pytest.approx(np.linspace(-1.0, 0.5, len(wav)), abs=1e-4)


@pytest.mark.parametrize(
    ("edited_text", "status"),
    [
        ("We go home now!", "no_change"),
        ("We go home now " + "again " * 400, "too_long"),  # 400 x 0.45 s inserted > 8192 frames
        # All 4 words replaced: [0.1, 1.45] s on a 146-frame condition, widened by 0.5 s, leaves no original frame.
        ("They went out.", "no_context"),
    ],
)
def test_synthesize_record_statuses(tmp_path: Path, edited_text: str, status: str) -> None:
    record = make_record(tmp_path, "single_span/insertion/u0", edited_text)
    model = FakeCFM()
    row = synthesize_record(record, model, FakeVocos(), str(tmp_path), **SAMPLING)
    assert row["status"] == status and row["synthetic_audio"] == "" and row["edit_regions"] == []
    assert row["original_duration"] == 2.0
    assert not model.sample_kwargs and not (tmp_path / "single_span").exists()


def test_synthesize_record_returns_no_row_for_a_missing_source_file(tmp_path: Path) -> None:
    # A row would resume as done and lose the record for good, e.g. after a run from the wrong directory.
    record = make_record(tmp_path, "single_span/deletion/u0", "We go home.")
    record["audio_path"] = str(tmp_path / "missing.mp3")
    model = FakeCFM()
    assert synthesize_record(record, model, FakeVocos(), str(tmp_path), **SAMPLING) is None
    assert not model.sample_kwargs and not (tmp_path / "single_span").exists()


def test_synthesize_record_raises_on_unreadable_audio_path(tmp_path: Path) -> None:
    # Only a missing file is skipped; other OS errors must surface instead of writing a row.
    record = make_record(tmp_path, "single_span/deletion/u0", "We go home.")
    record["audio_path"] = str(tmp_path / "clip.wav" / "x.mp3")
    with pytest.raises(NotADirectoryError):
        synthesize_record(record, FakeCFM(), FakeVocos(), str(tmp_path), **SAMPLING)


def test_cli_shards_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [make_record(tmp_path, f"single_span/deletion/u{i}", "We go home.") for i in (0, 4, 1, 5, 2)]
    records[-1]["audio_path"] = str(tmp_path / "missing.mp3")
    write_jsonl(tmp_path / "in.jsonl", records)
    output = tmp_path / "out.jsonl"
    write_jsonl(output, [{**records[0], "status": "ok"}])
    (tmp_path / "ckpt.safetensors").touch()
    monkeypatch.setattr(synthesize, "load_model", lambda *args, **kwargs: FakeCFM())
    monkeypatch.setattr(synthesize, "load_vocoder", lambda *args, **kwargs: FakeVocos())
    argv = ["synthesize", "--input", str(tmp_path / "in.jsonl"), "--output", str(output)]
    argv += ["--audio-dir", str(tmp_path / "audio"), "--ckpt", str(tmp_path / "ckpt.safetensors")]
    monkeypatch.setattr(sys, "argv", [*argv, "--vocoder", str(tmp_path), "--shard-id", "0", "--num-shards", "2"])
    # The run writes every other row, then fails so that run_sharded.sh merges nothing.
    with pytest.raises(FileNotFoundError, match="1 source audio files are missing"):
        synthesize.main()

    # Shard 0 of 2 holds the ids with an even crc32: u0 (already done), u1 and u2 (its audio file is missing).
    assert [zlib.crc32(record["id"].encode()) % 2 for record in records] == [0, 1, 0, 1, 0]
    rows = list(read_jsonl(str(output)))
    # u2 gets no row, so a re-run retries it instead of resuming past it.
    assert [row["id"] for row in rows] == ["single_span/deletion/u0", "single_span/deletion/u1"]
    assert rows[1]["synthetic_audio"] == str(tmp_path / "audio" / "single_span" / "deletion" / "u1.wav")
    assert [path.name for path in (tmp_path / "audio" / "single_span" / "deletion").iterdir()] == ["u1.wav"]


def test_cli_loads_the_downloaded_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_jsonl(tmp_path / "in.jsonl", [make_record(tmp_path, "single_span/deletion/u0", "We go home.")])
    loaded = []

    def load_model(model_cls: type, model_cfg: dict, ckpt_path: str, **kwargs: object) -> FakeCFM:
        loaded.append(ckpt_path)
        return FakeCFM()

    def load_vocoder(vocoder_name: str, is_local: bool, local_path: str, device: str) -> FakeVocos:
        loaded.append((is_local, local_path))
        return FakeVocos()

    monkeypatch.setattr(synthesize, "load_model", load_model)
    monkeypatch.setattr(synthesize, "load_vocoder", load_vocoder)
    argv = ["synthesize", "--input", "in.jsonl", "--output", "out.jsonl", "--audio-dir", "audio"]
    monkeypatch.setattr(sys, "argv", argv)
    f5_ckpt = tmp_path / "checkpoints" / "F5TTS_v1_Base" / "model_1250000.safetensors"
    with pytest.raises(FileNotFoundError, match=r"checkpoints/F5TTS_v1_Base/model_1250000\.safetensors not found"):
        synthesize.main()
    f5_ckpt.parent.mkdir(parents=True)
    f5_ckpt.touch()
    with pytest.raises(FileNotFoundError, match=r"checkpoints/vocos-mel-24khz not found; .*download_checkpoints\.sh"):
        synthesize.main()
    (tmp_path / "checkpoints" / "vocos-mel-24khz").mkdir()
    synthesize.main()
    assert loaded == ["checkpoints/F5TTS_v1_Base/model_1250000.safetensors", (True, "checkpoints/vocos-mel-24khz")]
    assert [row["status"] for row in read_jsonl("out.jsonl")] == ["ok"]
