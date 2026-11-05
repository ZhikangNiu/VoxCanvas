from pathlib import Path

import pytest

from voxcanvas.utils import check_checkpoints, check_uuids, rng_for


def test_rng_for_is_deterministic_and_independent() -> None:
    assert rng_for(0, "u1", "deletion").random() == rng_for(0, "u1", "deletion").random()
    assert rng_for(0, "u1", "deletion", "candidate").random() != rng_for(0, "u1", "deletion", "complete").random()


def test_check_uuids() -> None:
    check_uuids([{"uuid": "spk01_0001"}, {"uuid": "spk01_0002"}], "manifest.jsonl")
    for bad in ("", "spk01/0001", 7):
        with pytest.raises(ValueError, match="Invalid uuid .* in manifest.jsonl"):
            check_uuids([{"uuid": bad}], "manifest.jsonl")
    with pytest.raises(ValueError, match="Duplicate uuid 'spk01_0001' in manifest.jsonl"):
        check_uuids([{"uuid": "spk01_0001"}, {"uuid": "spk01_0002"}, {"uuid": "spk01_0001"}], "manifest.jsonl")


def test_check_checkpoints_checks_only_paths_under_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Hub ids, URLs and other paths are left to the loaders.
    check_checkpoints("Qwen/Qwen3-ASR-1.7B", "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors", "other/dir")
    with pytest.raises(FileNotFoundError, match=r"checkpoints/vocos-mel-24khz not found; .*download_checkpoints\.sh"):
        check_checkpoints("checkpoints/vocos-mel-24khz")
    (tmp_path / "checkpoints" / "vocos-mel-24khz").mkdir(parents=True)
    check_checkpoints("checkpoints/vocos-mel-24khz")
