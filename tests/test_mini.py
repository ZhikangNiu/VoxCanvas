import os
import re
import shutil
import subprocess
from pathlib import Path

import soundfile as sf

from voxcanvas.annotate import occurrence
from voxcanvas.annotate.records import SUBSETS_OF_TASK
from voxcanvas.text import gate, tokenize_content
from voxcanvas.utils import read_jsonl, rng_for


ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "examples" / "mini"
SCRIPT = ROOT / "scripts" / "run_mini_pipeline.sh"
SHARDED = ROOT / "scripts" / "run_sharded.sh"
DOWNLOAD = ROOT / "scripts" / "download_checkpoints.sh"
# Stage stub, called as --output <output_jsonl> --shard-id <shard_id> ...; shard FAIL_SHARD exits 3.
STUB_STAGE = (
    'output_jsonl=$2; shard_id=$4; if [ "$shard_id" = "${FAIL_SHARD:-}" ]; then exit 3; fi; '
    'echo "$shard_id $CUDA_VISIBLE_DEVICES" >> "$output_jsonl"'
)
# Downloader stub (hf, uvx) that logs its working directory, Hugging Face endpoint settings and argv.
STUB_DOWNLOADER = 'echo "$PWD|${HF_ENDPOINT:-}|${HF_HUB_DISABLE_IMPLICIT_TOKEN:-}|$(basename "$0") $*" >> "$STUB_LOG"'
EXPECTED_SUBSETS = {task: set(subsets) for task, subsets in SUBSETS_OF_TASK.items()}


def test_manifest_clips() -> None:
    rows = list(read_jsonl(str(MINI / "manifest.jsonl")))
    assert [row["language"] for row in rows].count("zh") == 10
    assert [row["language"] for row in rows].count("en") == 10
    assert len({row["uuid"] for row in rows}) == len(rows)
    # Emilia ids are <LANG>_B<shard>_S<speaker>_W<clip>: one clip per shard and speaker.
    assert len({row["uuid"][:9] for row in rows}) == len(rows)
    for row in rows:
        assert set(row) == {"uuid", "audio_path", "language"}
        assert row["uuid"].startswith(row["language"].upper() + "_B")
        assert row["audio_path"] == f"examples/mini/audio/{row['uuid']}.mp3"
        info = sf.info(str(ROOT / row["audio_path"]))
        assert info.channels == 1
        assert 4.0 <= info.duration <= 10.0
    assert {path.name for path in (MINI / "audio").iterdir()} == {f"{row['uuid']}.mp3" for row in rows}


def test_precomputed_aligned_rows() -> None:
    manifest = list(read_jsonl(str(MINI / "manifest.jsonl")))
    aligned = list(read_jsonl(str(MINI / "aligned.jsonl")))
    assert [{key: row[key] for key in ("uuid", "audio_path", "language")} for row in aligned] == manifest
    num_repeats = {"zh": 0, "en": 0}
    for row in aligned:
        assert gate(row)
        assert len(tokenize_content(row["text"], row["language"])) >= 10
        if occurrence.make_candidate(row, "d4", rng_for(0, row["uuid"])) is not None:
            num_repeats[row["language"]] += 1
    assert min(num_repeats.values()) >= 4


def test_readme_table_matches_manifest() -> None:
    readme = (MINI / "README.md").read_text(encoding="utf-8")
    table = re.findall(r"^\| ((?:ZH|EN)_\S+) \| (zh|en) \| ([\d.]+) \| (.+) \|$", readme, flags=re.MULTILINE)
    rows = {row["uuid"]: row for row in read_jsonl(str(MINI / "manifest.jsonl"))}
    aligned = {row["uuid"]: row["text"] for row in read_jsonl(str(MINI / "aligned.jsonl"))}
    assert [uuid for uuid, *_ in table] == list(rows)
    for uuid, language, duration, transcript in table:
        assert language == rows[uuid]["language"]
        assert abs(float(duration) - sf.info(str(ROOT / rows[uuid]["audio_path"])).duration) < 0.01
        assert transcript == aligned[uuid]
    assert "CC BY-NC 4.0" in readme
    assert "https://huggingface.co/datasets/amphion/Emilia-Dataset" in readme
    data_license = (MINI / "LICENSE").read_text(encoding="utf-8")
    assert "CC BY-NC 4.0" in data_license
    assert "https://creativecommons.org/licenses/by-nc/4.0/legalcode" in data_license


def test_pipeline_script() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    script = SCRIPT.read_text(encoding="utf-8")
    block = re.search(r"^task_subsets=\((.*?)^\)", script, flags=re.MULTILINE | re.DOTALL)
    subsets: dict[str, set[str]] = {}
    for task_subset in block.group(1).split():
        task, subset = task_subset.split(":")
        subsets.setdefault(task, set()).add(subset)
    assert subsets == EXPECTED_SUBSETS
    stages = re.findall(r"-m (voxcanvas\.[\w.]+)", script)
    assert stages == [
        "voxcanvas.align",
        "voxcanvas.annotate",
        "voxcanvas.synthesize",
        "voxcanvas.verify.asr",
        "voxcanvas.verify.sim",
        "voxcanvas.export",
    ]


def test_download_script(tmp_path: Path) -> None:
    subprocess.run(["bash", "-n", str(DOWNLOAD)], check=True)
    script = tmp_path / "scripts" / DOWNLOAD.name
    script.parent.mkdir()
    shutil.copy(DOWNLOAD, script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("hf", "uvx"):
        (bin_dir / name).write_text(f"#!/usr/bin/env bash\n{STUB_DOWNLOADER}\n")
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "log"
    # BASH_ENV may source an rc file that puts a real hf first on PATH.
    env = {key: value for key, value in os.environ.items() if not key.startswith("HF_") and key != "BASH_ENV"}
    env.update(PATH=f"{bin_dir}:{env['PATH']}", STUB_LOG=str(log))
    modelscope = "uvx --from modelscope-hub==0.6.1 modelscope download"
    vocos = "charactr/vocos-mel-24khz config.yaml pytorch_model.bin --local-dir checkpoints/vocos-mel-24khz"
    expected = {
        (): [
            "||hf download SWivid/F5-TTS F5TTS_v1_Base/model_1250000.safetensors --local-dir checkpoints",
            f"||hf download {vocos}",
            "||hf download Qwen/Qwen3-ASR-1.7B --local-dir checkpoints/Qwen3-ASR-1.7B",
            "||hf download Qwen/Qwen3-ForcedAligner-0.6B --local-dir checkpoints/Qwen3-ForcedAligner-0.6B",
        ],
        ("modelscope",): [
            f"||{modelscope} SWivid/F5-TTS_Emilia-ZH-EN model_1250000.safetensors"
            " --local-dir checkpoints/F5TTS_v1_Base",
            f"https://hf-mirror.com|1|hf download {vocos}",
            f"||{modelscope} Qwen/Qwen3-ASR-1.7B --local-dir checkpoints/Qwen3-ASR-1.7B",
            f"||{modelscope} Qwen/Qwen3-ForcedAligner-0.6B --local-dir checkpoints/Qwen3-ForcedAligner-0.6B",
        ],
    }
    for args, commands in expected.items():
        log.unlink(missing_ok=True)
        # Run from another directory: the downloads must still land in the repository's checkpoints/.
        run = subprocess.run(
            ["bash", str(script), *args], cwd=bin_dir, env=env, check=True, capture_output=True, text=True
        )
        assert log.read_text().splitlines() == [f"{tmp_path}|{command}" for command in commands]
        assert "to checkpoints/wavlm_large_finetune.pth" in run.stdout
    log.unlink()
    usage = subprocess.run(["bash", str(script), "huggingface"], env=env, capture_output=True, text=True)
    assert usage.returncode == 1
    assert "Usage: bash scripts/download_checkpoints.sh [hf|modelscope]" in usage.stderr
    assert not log.exists()


def test_sharded_launcher(tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"
    command = ["bash", str(SHARDED), "2", str(output), "bash", "-c", STUB_STAGE, "stage"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "4,5"}
    subprocess.run(command, check=True, env=env, capture_output=True)
    assert output.read_text() == "0 4\n1 5\n"
    output.unlink()
    failed = subprocess.run(command, env={**env, "FAIL_SHARD": "0"}, capture_output=True, text=True)
    assert failed.returncode == 1
    assert "Failed shards: 0;" in failed.stderr
    assert not output.exists()
    # A stage that leaves no shard file fails the merge, which must not truncate the previous output.
    previous = tmp_path / "previous.jsonl"
    previous.write_text("previous\n")
    no_shard_files = subprocess.run(["bash", str(SHARDED), "2", str(previous), "true"], env=env, capture_output=True)
    assert no_shard_files.returncode != 0
    assert previous.read_text() == "previous\n"
    # Requesting no shard, or hiding every GPU, must fail before any shard runs or the merge truncates the output.
    for num_gpus, visible, error in (("0", "4,5", "positive integer"), ("2", "", "lists 0 GPUs")):
        launch = subprocess.run(
            ["bash", str(SHARDED), num_gpus, str(previous), "bash", "-c", STUB_STAGE, "stage"],
            env={**env, "CUDA_VISIBLE_DEVICES": visible},
            capture_output=True,
            text=True,
        )
        assert launch.returncode == 1
        assert error in launch.stderr
        assert previous.read_text() == "previous\n"
