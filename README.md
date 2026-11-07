<div align="center">

# VoxCanvas Data Pipeline

**VoxCanvas-41M: A Large-Scale Bilingual Dataset for Instruction-Based Speech Editing and Its Learning Dynamics**

[![Demo](https://img.shields.io/badge/Demo-Page-blue?logo=googlechrome&logoColor=white)](https://zhikangniu.github.io/voxcanvas) [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.08251) [![Dataset](https://img.shields.io/badge/Dataset-VoxCanvas--41M-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/zkniu/VoxCanvas) [![Code License](https://img.shields.io/badge/Code_License-MIT-green)](LICENSE)

</div>

VoxCanvas-41M has about 41M Chinese and English speech-editing pairs, each a source utterance, a text instruction and the edited utterance.
This repository builds its **content-editing** part (single-span, multi-span and location-reference editing) from
real speech (Emilia) in five stages:

0. **align**: Qwen3-ASR-1.7B re-transcription and Qwen3-ForcedAligner-0.6B word timestamps;
1. **annotate**: rule-based edit selection, LLM-generated new words and instruction templates;
2. **synthesize**: F5-TTS masked infilling of the edited regions;
3. **verify**: ASR (zero WER / tone-pinyin error) and speaker-similarity checks;
4. **export**: the final pairs.

The algorithms, data formats and every difference from the released data are documented in
[docs/pipeline.md](docs/pipeline.md).

## 🔥 News

- **2026.11.07**: The paralinguistic (nonverbal) editing pipeline is released
  ([5fdb90e](https://github.com/ZhikangNiu/VoxCanvas/commit/5fdb90eae4d60925e6e82c422d2da9957dccb372)); see
  [Nonverbal editing](#nonverbal-editing).
- **2026.11.07**: [VoxCanvas-41M](https://huggingface.co/datasets/zkniu/VoxCanvas) is fully uploaded, including the
  paralinguistic subset.
- **2026.11.05**: We release the speech-editing data pipeline; the dataset is being uploaded.

## Repository layout

```
src/voxcanvas/
  align.py                 stage 0: ASR transcript + forced alignment
  annotate/                stage 1 (python -m voxcanvas.annotate)
    __main__.py            CLI: gate -> candidates -> sampling -> LLM completion -> records
    content.py             single_span (deletion, insertion, substitution) and multi_span
    occurrence.py          repeated words: d1 d4 m1 m2 i1 i2
    boundary.py            sentence head / tail: l_d l_m l_i
    timestamp.py           time point / interval: l_t_d l_t_m l_t_i
    nonverbal.py           nonverbal events: delete every [tag] of one type
    templates.py           instruction templates (zh / en)
    prompts.py             LLM prompts
    llm.py                 OpenAI-compatible client, LLM replacement / insertion generation
    records.py             edit record, length-stratified sampling
  text.py                  tokenizers, alignment gate, edit application, validators
  synthesize.py            stage 2: F5-TTS edit synthesis
  verify/asr.py            stage 3a: ASR check
  verify/sim.py            stage 3b: speaker similarity
  export.py                stage 4: filtering into final pairs
  utils.py                 jsonl helpers, shard selection and resume, seeded random streams
scripts/
  download_checkpoints.sh  model checkpoints from Hugging Face or ModelScope
  run_mini_pipeline.sh     all stages on the mini set, one GPU
  run_mini_nonverbal.sh    nonverbal editing on the mini nonverbal set, one GPU
  run_sharded.sh           one process per GPU for a GPU stage, then merge
checkpoints/               downloaded model checkpoints (git-ignored)
examples/mini/             20 Emilia clips (10 zh, 10 en), manifest and precomputed alignment
examples/mini_nonverbal/   20 NonVerbalSpeech-38K clips with tagged transcripts
docs/pipeline.md           algorithms, formats, differences from the release
tests/                     pytest suite
third_party/F5-TTS         F5-TTS v1.1.20 (git submodule)
```

## Installation

Requirements:
- Linux with an NVIDIA GPU (CUDA 12)
- [uv](https://docs.astral.sh/uv/) manage environment
- FFmpeg < 7 (`apt install ffmpeg`), used by torchaudio to decode mp3

```bash
git clone --recursive https://github.com/ZhikangNiu/VoxCanvas.git
cd VoxCanvas
uv sync --locked
source .venv/bin/activate
```

### Checkpoints

Download the checkpoints into `checkpoints/`, from Hugging Face or from ModelScope (models that ModelScope does not
host come from hf-mirror.com):

```bash
bash scripts/download_checkpoints.sh              # Hugging Face
bash scripts/download_checkpoints.sh modelscope   # ModelScope
```

Then download `wavlm_large_finetune.pth` by hand from
[Google Drive](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view) to
`checkpoints/wavlm_large_finetune.pth`.

| model | used by | path in `checkpoints/` |
|---|---|---|
| F5TTS_v1_Base | synthesize | `F5TTS_v1_Base/model_1250000.safetensors` |
| Vocos | synthesize | `vocos-mel-24khz/` |
| Qwen3-ASR-1.7B | align, verify.asr | `Qwen3-ASR-1.7B/` |
| Qwen3-ForcedAligner-0.6B | align | `Qwen3-ForcedAligner-0.6B/` |
| WavLM-Large speaker verification | verify.sim | `wavlm_large_finetune.pth` |

### LLM endpoint

`annotate` works with any OpenAI-compatible chat API (deletion subsets need no LLM). The default is DeepSeek; put the
key in `.env`:

```bash
cp .env.example .env          # then edit: DEEPSEEK_API_KEY=sk-...
```

For another endpoint, e.g. a self-hosted Qwen3.6-35B-A3B on vLLM (the LLM used for the release), pass
`--base-url`, `--model`, `--api-key-env` and optionally `--extra-body`:

```bash
python -m voxcanvas.annotate ... --base-url http://localhost:8000/v1 --model qwen3.6-35b-a3b \
    --api-key-env LOCAL_LLM_KEY --extra-body '{"top_k": 20, "chat_template_kwargs": {"enable_thinking": false}}'
```

## Quick start: the mini set

`examples/mini/` holds 20 Emilia clips (10 zh, 10 en, 135 s in total; see its [README](examples/mini/README.md)).
`scripts/run_mini_pipeline.sh` runs every stage on them into `outputs/mini/` on the first visible GPU:

```bash
# all stages, including align on the GPU
bash scripts/run_mini_pipeline.sh

# skip align and start from the precomputed examples/mini/aligned.jsonl
SKIP_ALIGN=1 bash scripts/run_mini_pipeline.sh
```

Run it from the repository root with the environment active (or prefix with `uv run`). Checkpoints default to
`checkpoints/` ([Checkpoints](#checkpoints)). Set `F5_CKPT`, `VOCODER`, `ASR_MODEL`, `ALIGNER_MODEL` and `SIM_CKPT` to
use other paths, and `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY_ENV` and `LLM_EXTRA_BODY` for another LLM; unset
variables keep the stage defaults (see [examples/mini/README.md](examples/mini/README.md#running-the-pipeline)). Every
stage resumes, so the script can be re-run after an interruption. The final pairs are in `outputs/mini/pairs.jsonl`, and `export` logs a table of kept /
dropped pairs per subset and language.

On one (shared) H20 GPU with DeepSeek, the full run took 6.5 min:

| stage | rows in -> out | time |
|---|---|---|
| align | 20 clips -> 20 aligned rows | 27 s |
| annotate (16 subsets) | 20 rows -> 287 edit records | 57 s |
| synthesize | 287 -> 287 ok | 184 s |
| verify.asr | 287 | 46 s |
| verify.sim | 287 | 52 s |
| export | 287 -> 269 pairs (14 ASR mismatches, 4 low SIM) | < 1 s |

These times include model loading. LLM answers vary, so counts differ slightly between runs.

The rule-based subsets need neither a GPU nor an API key, which is a quick way to see the annotation:

```bash
python -m voxcanvas.annotate --task location_reference --subset d4 \
    --input examples/mini/aligned.jsonl --output outputs/try/d4.jsonl
```

### Examples from the mini run

| subset | lang | instruction | original_text -> synthetic_text |
|---|---|---|---|
| single_span / substitution | en | Replace 'draw' with 'paint'. *(reverse: Replace 'paint' with 'draw'.)* | And apparently, I'm supposed to draw something that makes me happy or reminds me of happiness. -> And apparently, I'm supposed to paint something that makes me happy or reminds me of happiness. |
| single_span / insertion | zh | 在‘产生了’后面加上‘一丝’ *(reverse: 删除‘一丝’)* | 皇上对太子已经产生了猜忌之心，太子的日子好过不了了。 -> 皇上对太子已经产生了一丝猜忌之心，太子的日子好过不了了。 |
| multi_span / multi | zh | 在‘梦想’后面加上‘并且坚持’，在‘但是’后面加上‘老实说’，在‘的冒险’后面加上‘其实’ *(reverse: 删除‘其实’，删除‘并且坚持’，删掉‘老实说’)* | 追求梦想是要冒险的，但是人生最大的冒险就是不去冒险。 -> 追求梦想并且坚持是要冒险的，但是老实说人生最大的冒险其实就是不去冒险。 |
| location_reference / d4 | en | Please delete the second 'very'. | Books about the law are notorious for being very long, very dull, and very difficult to read. -> Books about the law are notorious for being very long, dull, and very difficult to read. |
| location_reference / m1 | zh | 把全部的‘妈妈’替换成‘奶奶’ | 妈妈以前不是这样的。小凯的妈妈很漂亮，说话也很温柔。小凯非常喜欢妈妈。 -> 奶奶以前不是这样的。小凯的奶奶很漂亮，说话也很温柔。小凯非常喜欢奶奶。 |
| location_reference / l_d | zh | 删除句首的‘因此啊’ | 因此啊，在贪婪之前，我们必须经过思考，确保我们的贪婪不会让我们堕入深渊。 -> 在贪婪之前，我们必须经过思考，确保我们的贪婪不会让我们堕入深渊。 |
| location_reference / l_t_m | en | Change "belief" from 7.8s to 8.2s into "imagination". | And of course, the force in Star Wars makes its successful users strong beyond belief. -> And of course, the force in Star Wars makes its successful users strong beyond imagination. |

Single-span and multi-span pairs are used in the reverse direction for training (edited audio -> original audio, with
the reverse instruction); location-reference pairs are used forward. See [docs/pipeline.md](docs/pipeline.md#2-subsets-and-paper-table-1).

## Using your own data

Write a manifest, a jsonl file with one row per clip:

```json
{"uuid": "spk01_0001", "audio_path": "/data/my_corpus/spk01_0001.wav", "language": "zh"}
```

- `uuid`: unique, without `/`; it is the last part of every record id.
- `audio_path`: any format torchaudio reads.
- `language`: `zh` or `en`.

Clips should be single-speaker utterances of 3-30 s, like Emilia's; the pipeline does not segment long recordings.
Then run all stages on one GPU:

```bash
MANIFEST_JSONL=data/my_manifest.jsonl OUTPUT_DIR=outputs/my_corpus bash scripts/run_mini_pipeline.sh
```

If you already have aligned rows (manifest fields plus `text` and `align_word`, see
[docs/pipeline.md §8](docs/pipeline.md#8-data-formats)), skip `align` with
`SKIP_ALIGN=1 ALIGNED_JSONL=data/my_aligned.jsonl`.

### Nonverbal editing

Add the tagged transcript to each manifest row (`"transcript": "那是[sniff]传说中的此莲真火。"`) and run
`scripts/run_mini_nonverbal.sh` with `MANIFEST_JSONL` / `OUTPUT_DIR`; `examples/mini_nonverbal/` is a ready-made input.
See [docs/pipeline.md §4.8](docs/pipeline.md#48-nonverbal).

## Running at scale

VoxCanvas-41M was built from the ZH and EN subsets of the original
[Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) release (Hugging Face revision `fc71e07e`); the
released pairs are a selected subset of what the pipeline produces. Process a large corpus one manifest shard at a
time. `scripts/run_sharded.sh <num_gpus> <output> <command>` runs a GPU stage on every GPU of a node and merges the
outputs; keep `<num_gpus>` fixed when re-running a stage, so it resumes.

```bash
manifest_jsonl=data/manifests/ZH_B00000.jsonl   # e.g. one Emilia shard: uuid = id, audio_path = wav, language
shard_name=ZH_B00000

bash scripts/run_sharded.sh 8 data/aligned/$shard_name.jsonl python -m voxcanvas.align --input $manifest_jsonl
for task_subset in single_span:deletion single_span:insertion multi_span:multi location_reference:d4; do   # any subsets
    python -m voxcanvas.annotate --task ${task_subset%%:*} --subset ${task_subset#*:} --concurrency 64 \
        --input data/aligned/$shard_name.jsonl --output data/edits/$shard_name/${task_subset/:/_}.jsonl
done
cat data/edits/$shard_name/*.jsonl > data/edits/$shard_name.jsonl
bash scripts/run_sharded.sh 8 data/synth/$shard_name.jsonl python -m voxcanvas.synthesize \
    --input data/edits/$shard_name.jsonl --audio-dir data/audio/$shard_name
bash scripts/run_sharded.sh 8 data/asr/$shard_name.jsonl python -m voxcanvas.verify.asr \
    --input data/synth/$shard_name.jsonl
bash scripts/run_sharded.sh 8 data/sim/$shard_name.jsonl python -m voxcanvas.verify.sim \
    --input data/asr/$shard_name.jsonl
python -m voxcanvas.export --input data/sim/$shard_name.jsonl --output data/pairs/$shard_name.jsonl
```

Resuming, re-sharding and throughput are covered in [docs/pipeline.md §1](docs/pipeline.md#1-overview).

## Development

```bash
.venv/bin/pre-commit install   # once: ruff check / format on every commit (same ruff version as the dev group)
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python -m pytest -q    # runs on CPU; the SIM CLI test is skipped without a CUDA GPU
```

[CI](.github/workflows/ci.yml) runs the same lint and test checks on every push and pull request.

## Citation

```bibtex
@article{niu2026voxcanvas,
  title        = {VoxCanvas-41M: A Large-Scale Bilingual Dataset for Instruction-Based Speech Editing and Its Learning Dynamics},
  author       = {Niu, Zhikang and Tu, Wenming and Li, Yalin and Huo, Yanru and Liu, Junxi and Wang, Tianrui and Yan, Ruiqi
                  and Liang, Junxuan and Zheng, Zilong and Ma, Ziyang and Chen, Xie},
  journal      = {arXiv preprint arXiv:2611.08251},
  year         = {2026}
}
```

This pipeline builds on [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset),
[F5-TTS](https://github.com/SWivid/F5-TTS) and [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B); please cite them
as well.

## License

The code is released under the [MIT License](LICENSE). Data, including `examples/mini/`, `examples/mini_nonverbal/`
and anything built with this pipeline, follows the license of its source corpus (Emilia, NonVerbalSpeech-38K:
CC BY-NC 4.0).
