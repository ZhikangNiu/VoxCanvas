
# VoxCanvas Content-Editing Pipeline

This document specifies the data construction pipeline for the content-editing subsets of **VoxCanvas-41M** (single-span, multi-span, location-reference, and paralinguistic editing on the Chinese and English subsets of Emilia). It covers the end-to-end architecture, stage-by-stage algorithms, data schemas, and implementation differences relative to the released dataset.

---

## 1. Architecture Overview

```text
manifest.jsonl           {uuid, audio_path, language}
    │
    ├─ 0. voxcanvas.align          Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B          [GPU]
    ▼
aligned.jsonl            + text, align_word
    │
    ├─ 1. voxcanvas.annotate       Gate -> Rule Selection -> LLM -> Templates         [CPU + LLM API]
    ▼
edits.jsonl              + original/edited_text, forward/reverse_instruction, edits
    │
    ├─ 2. voxcanvas.synthesize     F5-TTS Masked Infilling + Vocos                    [GPU]
    ▼
synth.jsonl + wavs       + status, synthetic_audio, durations, edit_regions
    │
    ├─ 3a. voxcanvas.verify.asr    Qwen3-ASR-1.7B (en: WER, zh: Tone-Pinyin PER)      [GPU]
    ▼
asr.jsonl                + asr_text, asr_score
    │
    ├─ 3b. voxcanvas.verify.sim    WavLM-Large ECAPA-TDNN Cosine Similarity           [GPU]
    ▼
sim.jsonl                + sim
    │
    ├─ 4. voxcanvas.export         Filter: status=="ok", dur<=30s, asr==0, sim>0.5    [CPU]
    ▼
pairs.jsonl              Final training pairs (release schema, path-based audio)
```

| Stage | Paper Ref | Source Modules |
| --- | --- | --- |
| `align` | §3.2 | `src/voxcanvas/align.py` |
| `annotate` | §3.3, App. A | `src/voxcanvas/annotate/` (`content.py`, `occurrence.py`, `boundary.py`, `timestamp.py`, `templates.py`, `prompts.py`, `llm.py`, `records.py`) & `src/voxcanvas/text.py` |
| `synthesize` | §3.4 | `src/voxcanvas/synthesize.py` |
| `verify` | §3.5 | `src/voxcanvas/verify/asr.py`, `src/voxcanvas/verify/sim.py` |
| `export` | §3.5 | `src/voxcanvas/export.py` |

### 1.1 Execution Model, Resumption & Sharding

Every stage runs as a standalone CLI (`python -m voxcanvas.<stage> --help`) transforming one JSONL file into another:

- **Checkpointing & Resumption**: All stages except `export` operate in append mode and skip existing record IDs (`uuid` for `align`, `id` for downstream stages). Incomplete trailing lines from interrupted runs are automatically truncated upon restart.
- **Deterministic Sharding**: GPU stages (`align`, `synthesize`, `verify.asr`, `verify.sim`) support `--shard-id` and `--num-shards`. Work is partitioned via `zlib.crc32(key) % num_shards == shard_id`, ensuring shard assignment remains invariant to input ordering or dataset growth. Keep `--num-shards` constant across resumptions to avoid duplicate computations.
- **Multi-GPU Orchestration**: `scripts/run_sharded.sh` spawns one worker per visible GPU, awaits completion, and atomically concatenates shard outputs in shard order. If any worker exits non-zero, merging is aborted to preserve previous outputs.

### 1.2 Checkpoints

By default, stages load weights from `checkpoints/` relative to the repository root. Run `scripts/download_checkpoints.sh [hf|modelscope]` to fetch all required weights except WavLM:

| Path in `checkpoints/` | Size | Hugging Face (`hf`) | ModelScope (`modelscope`) | SHA256 Prefix |
| --- | --- | --- | --- | --- |
| `F5TTS_v1_Base/model_1250000.safetensors` | 1.35 GB | `SWivid/F5-TTS` | `SWivid/F5-TTS_Emilia-ZH-EN` | `670900fd` |
| `vocos-mel-24khz/config.yaml` | — | `charactr/vocos-mel-24khz` | via `hf-mirror.com` | `da903392` |
| `vocos-mel-24khz/pytorch_model.bin` | 54 MB | `charactr/vocos-mel-24khz` | via `hf-mirror.com` | `97ec976a` |
| `Qwen3-ASR-1.7B/` | 4.7 GB | `Qwen/Qwen3-ASR-1.7B` | `Qwen/Qwen3-ASR-1.7B` | `a4cd1f1a` / `6e0b9d9e` |
| `Qwen3-ForcedAligner-0.6B/` | 1.8 GB | `Qwen/Qwen3-ForcedAligner-0.6B` | `Qwen/Qwen3-ForcedAligner-0.6B` | `47831d0e` |
| `wavlm_large_finetune.pth` | 1.3 GB | [Manual Download (Google Drive)](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view) | Same | `51f07e3b` |

- **Vocab & Auxiliary Weights**: `synthesize` uses the `vocab.txt` bundled with `f5_tts`. `verify.sim` additionally requires `s3prl`'s WavLM-Large (`wavlm_large.pt`, 1.26 GB), fetched automatically via `torch.hub` to `~/.cache/s3prl/download`.

### 1.3 Throughput & Resource Profile

Measured per worker process on a single NVIDIA H20 GPU with default hyperparameters (excluding model initialization):

| Stage | Throughput | Peak VRAM | Notes |
| --- | --- | --- | --- |
| `align` | ~7.5 clips/s | ~15.0 GiB | `batch_size=32` (~1,400 GPU-hours for 38.1M Emilia clips) |
| `synthesize` | ~1.5–1.8 edits/s | ~2.5 GiB | `nfe_step=32` |
| `verify.asr` | ~7.0–9.0 edits/s | ~11.0 GiB | `batch_size=32` |
| `verify.sim` | ~10.0 edits/s | ~3.0 GiB | `batch_size=1` |

> **Note**: Both `align` and `verify.asr` use the `transformers` backend of `qwen-asr` to maintain environment compatibility with `f5_tts`. For large-scale production, switching to the `vLLM` backend in an isolated environment yields a multi-fold speedup.

---

## 2. Tasks, Subsets & Training Directions

Because F5-TTS performs masked infilling, synthetic speech can exhibit slight prosodic or acoustic artifacts at edit boundaries. To prevent the edited model from learning synthesis artifacts as generation targets, **single-span and multi-span tasks invert the training direction** (Source = Synthetic Audio, Target = Real Audio, Instruction = Reverse Instruction). Location-reference tasks require referencing positions in the source utterance and are therefore trained in the **forward direction**.

| Paper Table 1 Row | `task` | `subset` | Forward Operation (Real $\to$ Synth) | LLM | Trained Direction | Released Scale |
| --- | --- | --- | --- | --- | --- | --- |
| Single-span insertion | `single_span` | `deletion` | Delete a unique span of 1–6 words | No | **Reversed** | ~10M |
| Single-span deletion | `single_span` | `insertion` | Insert new words adjacent to a unique anchor | Yes | **Reversed** | ~10M |
| Single-span substitution | `single_span` | `substitution` | Replace a unique span of 1–6 words | Yes | **Reversed** | ~10M |
| Multi-span (2–6 edits) | `multi_span` | `multi` | 2–6 separated deletions / substitutions / insertions | Yes | **Reversed** | ~10M |
| Location: ordinal | `location_reference` | `d4`, `m2`, `i2` | Delete / replace / insert at the $i$-th occurrence of a repeat | `m2`, `i2` | **Forward** | 100K / row |
| Location: all occurrences | `location_reference` | `d1`, `m1`, `i1` | Delete / replace / insert at every occurrence of a repeat | `m1`, `i1` | **Forward** | 100K / row |
| Location: boundary | `location_reference` | `l_d`, `l_m`, `l_i` | Delete 1–6 words / replace / insert at utterance head or tail | `l_m`, `l_i` | **Forward** | 100K / row |
| Location: timestamp | `location_reference` | `l_t_d`, `l_t_m`, `l_t_i` | Delete / replace / insert at a timestamp or interval | `l_t_m`, `l_t_i` | **Forward** | 100K / row |
| Paralinguistic | `nonverbal` | `insertion` | Delete all `[tag]` instances of one nonverbal event type | No | **Reversed** | — |

---

## 3. Stage 0: Forced Alignment (`voxcanvas.align`)

### Precomputed Alignments

The ASR transcripts and forced-alignment timestamps have already been released in the [VoxCanvas Hugging Face repository](https://huggingface.co/datasets/zkniu/VoxCanvas/tree/main/data/refine_emilia_text). These annotations cover six languages from Emilia (zh, en, ja, fr, ko, de), including word- or character-level timestamps.

**Researchers using the released annotations can skip Stage 0** and proceed directly to Stage 1 after preparing the aligned input records. Stage 0 remains available for full pipeline reproduction or processing new speech corpora.

The precomputed annotations can be downloaded using:

```bash
hf download zkniu/VoxCanvas \
  --repo-type dataset \
  --include "data/refine_emilia_text/**" \
  --local-dir ./voxcanvas_text
```

The released files contain `id`, `wav`, `asr_text`, `align_word`, and `duration`, without audio. To use them with Stage 1, the records should be converted to the pipeline's `aligned.jsonl` schema, with audio paths resolved against the original Emilia corpus.

```bash
python -m voxcanvas.align \
  --input manifest.jsonl \
  --output aligned.jsonl \
  [--asr-model checkpoints/Qwen3-ASR-1.7B] \
  [--aligner-model checkpoints/Qwen3-ForcedAligner-0.6B] \
  [--batch-size 32] [--max-new-tokens 1024]
```

- **Re-transcription**: Rather than relying on original Emilia transcripts (which have only 27% EN and 63% ZH strict token agreement with ASR), every clip is re-transcribed with `Qwen3-ASR-1.7B` and aligned via `Qwen3-ForcedAligner-0.6B` in `bf16` using the manifest's forced language tag (`zh` $\to$ Chinese, `en` $\to$ English).
- **Granularity**: `align_word` outputs `[[token, start_s, end_s], ...]` on an 80 ms grid—one entry per CJK character (`zh`) or word (`en`), preserving original case.
- **Fault Tolerance**: Failed batches (e.g., CUDA OOM or corrupt audio) fall back to single-clip retries. Clips that fail decoding or yield empty transcripts/timestamps are skipped and logged.

---

## 4. Stage 1: Edit Annotation (`voxcanvas.annotate`)

```bash
python -m voxcanvas.annotate \
  --task {single_span,multi_span,location_reference,nonverbal} \
  --subset <SUBSET> \
  --input aligned.jsonl \
  --output edits.jsonl \
  [--num-samples 0] [--seed 0] [--concurrency 16] \
  [--base-url https://api.deepseek.com] [--model deepseek-chat] \
  [--api-key-env DEEPSEEK_API_KEY] [--extra-body JSON]
```

### 4.1 Pipeline Execution & Text Primitives

Annotation executes four sequential steps per utterance:

1. **Gate (`text.gate`)**: Filters out utterances with < 4 content tokens, `[UNK]` markers, foreign scripts (`en` rejects CJK/kana/hangul; `zh` rejects kana/hangul and embedded Latin letters), or token-granularity mismatches between `text` and `align_word`. It also rejects compound tokens where one aligner token spans multiple content tokens (e.g., `U.S.`, `5:00`, `1,000`), guaranteeing a 1-to-1 index mapping (`align_word[k]`) for downstream synthesis.
2. **Candidate Generation (`make_candidate`)**: Rule-based selection of target spans/anchors without LLM calls.
3. **Stratified Sampling (`records.stratified_sample`)**: Active when `--num-samples N > 0`. Candidates are grouped into length buckets and sampled round-robin from longest to shortest using deterministic per-row priority streams.
4. **LLM Completion & Validation (`complete_candidate`)**: Concurrent worker threads query the LLM, apply character-span edits, enforce linguistic/reversibility constraints, and render templates.

**Core Text Primitives (`src/voxcanvas/text.py`):**

- **Content Tokenization (`tokenize_content`)**: `zh` segments via `jieba` (splitting embedded alphanumeric runs via regex); `en` matches `[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*`. Punctuation acts as segment boundaries.
- **Edit Application (`apply_edits`)**: Non-overlapping character-span mutations applied right-to-left. Manages English whitespace normalization and collapses adjacent punctuation runs resulting from deletions.
- **Alignment Preservation (`edit_keeps_alignment`)**: Rejects edits that alter aligner tokenization outside the edited span (e.g., deleting `起` in `21起500` merges `21500` into a single token; deleting `he` in `thinking he'` leaves an orphan apostrophe `thinking'`).
- **Reversibility (`content_key`)**: Checks that applying the inverse operation restores the exact lowercased alphanumeric/CJK character sequence of the original text.
- **Deterministic RNG (`rng_for`)**: Separate `random.Random` streams keyed by `(seed, uuid, subset, phase)` ensure candidate selection, sampling, and completion are invariant to input order and concurrency.

### 4.2 LLM Generation & Validation

- **Client**: Connects to any OpenAI-compatible endpoint (`temperature=1.0`, `top_p=0.95`, `max_tokens=256`). Strips `<think>...</think>` blocks and leading labels (`Answer:`, `输出：`).
- **Rejection Sampling**: Up to 5 draws per candidate. A draw is accepted only if it passes surface validation (`is_valid_replacement` / `is_valid_insertion`):
  - Non-empty, matching target language, no leading/trailing punctuation or symbols.
  - **Replacements**: Must not already appear in the sentence (case-insensitive) and cannot be a pure Chinese homophone/near-homophone swap (`他她它`, `的地得`, `在再`, `做作`, `座坐`, `那哪`, `了啦`, `吗嘛`, `呢呐`, `和合`).
  - **Insertions**: Must not contain the anchor or duplicate adjacent words.
- **Edit-Level Acceptance**: Must change the spoken content key, introduce no new adjacent word/character repetitions (`has_adjacent_duplicate`), and remain strictly reversible.

### 4.3 Task Specifications

- **Single-Span (`content.py`)**:
  - **Deletion / Substitution**: Uniformly picks a token index and expands a span of $k \sim U\{1..\min(6, n-\text{idx})\}$ tokens (leaving $\ge 2$ tokens for deletion) alternately left and right until unique in the sentence (capped at 60% of sentence length).
  - **Insertion**: Picks direction $\sim U\{\text{before}, \text{after}\}$ and grows the anchor away from the insertion point within the punctuation segment until unique.
  - **Reverse Instructions**: Swaps $X \leftrightarrow Y$ for substitution. For insertion, emits bare deletion (`Delete the 'Y'.` / `删除‘Y’`) if $Y$ is unique in the edited text, or anchored deletion (`Delete the 'Y' before/after 'X'.`) otherwise. For deletion, grows a unique whole-word context adjacent to the deletion point to form an `Add 'X' before/after 'Context'` instruction.

- **Multi-Span (`content.py`)**:
  - Places $k \sim U\{2..6\}$ non-overlapping edits separated by at least 2 tokens (weights: `del:sub:ins_before:ins_after = 2:2:1:1`).
  - Queries the LLM independently per substitution/insertion region.
  - Forward instructions join sub-instructions in text order (`" "` for `en`, `"，"` for `zh`); reverse instructions shuffle sub-instruction order and verify that right-to-left replay restores the original content key.

- **Occurrences (`occurrence.py` — `d1, d4, m1, m2, i1, i2`)**:
  - Extracts maximal non-overlapping repeated token n-grams within punctuation segments (`find_repeats`), picking the longest repeat (up to 8 occurrences for ordinal subsets `d4/m2/i2`).
  - `d4/m2/i2` target the $i$-th occurrence ($i \sim U\{1..\text{count}\}$); `d1/m1/i1` target all occurrences.

- **Sentence Boundary (`boundary.py` — `l_d, l_m, l_i`)**:
  - Operates on the first (`head`) or last (`tail`) tokens of the entire transcript (`l_d` deletes 1–6 edge tokens within the edge punctuation segment; `l_m` replaces the single edge token; `l_i` inserts before/after it).

- **Timestamps (`timestamp.py` — `l_t_d, l_t_m, l_t_i`)**:
  - Operates in `point` mode (single word located by midpoint $t$) or `span` mode (1–5 consecutive words located by $[t_{\text{start}}, t_{\text{end}}]$, rounded to $0.1\text{ s}$).
  - **Disambiguation Check**: Drops candidates if the rounded timestamps do not uniquely resolve back to the exact target words. Includes the surface word in the instruction with probability $0.5$ (`has_word ~ Bernoulli(0.5)`).

- **Paralinguistic (`nonverbal`)**:
  - Requires pre-tagged transcripts (`[laugh]`, `[sigh]`, etc.). Aligns tag-free text via `align --use-transcript`, deletes all instances of a single tag type in the forward pass, and trains in the reverse direction (`Add '[tag]' before/after 'Anchor'`).

---

## 5. Stage 2: Speech Synthesis (`voxcanvas.synthesize`)

```bash
python -m voxcanvas.synthesize \
  --input edits.jsonl \
  --output synth.jsonl \
  --audio-dir outputs/audio \
  [--ckpt checkpoints/F5TTS_v1_Base/model_1250000.safetensors] \
  [--vocoder checkpoints/vocos-mel-24khz] \
  [--nfe-step 32] [--cfg-strength 2.0] [--sway-sampling-coef -1.0] [--seed 0]
```

Edited speech is synthesized via **F5-TTS masked infilling** (`F5TTS_v1_Base` + `Vocos` 24 kHz vocoder):

1. **Preprocessing**: Audio is downmixed to mono, resampled to 24 kHz, RMS-normalized (boosted to 0.1 if RMS $< 0.1$), and converted to 100-bin Vocos log-mel spectrograms (hop size 256, $93.75\text{ frames/s}$).

2. **Edit Plan (`build_edit_plan`)**: Runs `difflib.SequenceMatcher` between `align_word` tokens and `tokenize_aligned(edited_text)`. Local speech rate ($s_{\text{tok}}$) is estimated from aligned tokens and clamped to $[0.08, 0.32]\text{ s}$ (`zh`) or $[0.10, 0.45]\text{ s}$ (`en`). Target durations per diff opcode:
   - **Replace**: $\max(0.18\text{ s}, d_{\text{orig}} \cdot \frac{n_{\text{new}}}{n_{\text{old}}})$
   - **Insert**: $\max(0.18\text{ s}, n_{\text{new}} \cdot s_{\text{tok}})$, inserted into the inter-word pause.
   - **Delete**: Replaces deleted words with $0.10\text{ s}$ of generated acoustic filler.

   Adjacent/overlapping edit spans are merged and projected onto the synthetic timeline.

3. **Conditioning & Masking**: Edited spans in the source mel are replaced with zero frames of the target duration. The generation mask covers the edited spans **widened by $\pm 0.5\text{ s}$** on both sides to ensure smooth prosodic transitions.

4. **CFM Sampling & Vocoding**: Sampled via Flow Matching (`steps=32`, `cfg=2.0`, `sway=-1.0`) conditioned on full edited pinyin/text with a deterministic per-record seed. The entire mel is decoded by Vocos (no waveform splicing), RMS scaling is inverted, and peaks $> 1.0$ are normalized before saving as 24 kHz 16-bit PCM WAV.

5. **Status Filtering**: Records are marked `ok` or skipped with explicit status flags:
   - `no_change`: Zero spoken token delta.
   - `too_long`: Condition exceeds the DiT context limit of 8,192 frames (~87 s).
   - `no_context`: The $\pm 0.5\text{ s}$ widened mask covers 100% of the utterance, leaving no unmasked acoustic prompt to preserve speaker identity.

---

## 6. Stage 3 & 4: Verification & Export (`verify` & `export`)

```bash
# Stage 3a: Intelligibility Verification (ASR)
python -m voxcanvas.verify.asr --input synth.jsonl --output asr.jsonl [--batch-size 32]

# Stage 3b: Speaker Similarity Verification (SIM)
python -m voxcanvas.verify.sim --input asr.jsonl --output sim.jsonl [--ckpt checkpoints/wavlm_large_finetune.pth]

# Stage 4: Filtering & Export
python -m voxcanvas.export --input sim.jsonl --output pairs.jsonl [--max-duration 30] [--max-asr-score 0] [--min-sim 0.5]
```

- **Intelligibility (`verify.asr`)**: Transcribes synthetic audio via `Qwen3-ASR-1.7B` without language hints and computes `asr_score` against `edited_text`:
  - **English (WER)**: Lowercased, punctuation stripped, word-level edit distance.
  - **Chinese (Tone-Pinyin PER)**: Punctuation stripped, converted via `pypinyin` (`TONE3`, neutral tone as `5`), evaluated as syllable-level edit distance so homophonous characters sharing the same tone are not penalized.
- **Speaker Consistency (`verify.sim`)**: Resamples both original and synthetic audio to 16 kHz and computes full-utterance cosine similarity (`sim`) using `ECAPA_TDNN_SMALL` backed by UniSpeech `WavLM-Large`.
- **Export Criteria (`export`)**: Retains records satisfying:

$$
\text{status} == \text{"ok"} \quad \land \quad \text{synthetic\_duration} \le 30\text{ s} \quad \land \quad \text{asr\_score} == 0 \quad \land \quad \text{sim} > 0.5
$$

---

## 7. Data Schemas

All files use UTF-8 JSON Lines (`.jsonl`). Relative audio paths resolve from the repository root.

### 7.1 Input Manifest (`manifest.jsonl`) & Aligned Output (`aligned.jsonl`)

```json
{
  "uuid": "EN_B00004_S05804_W000003",
  "audio_path": "examples/mini/audio/EN_B00004_S05804_W000003.mp3",
  "language": "en",
  "text": "Books about the law are notorious for being very long, very dull, and very difficult to read.",
  "align_word": [["Books", 0.0, 0.32], ["about", 0.32, 0.64], ["the", 0.64, 0.64]]
}
```

### 7.2 Edit Record (`edits.jsonl`)

```json
{
  "id": "location_reference/d4/EN_B00004_S05804_W000003",
  "task": "location_reference",
  "subset": "d4",
  "edit_type": "deletion",
  "language": "en",
  "uuid": "EN_B00004_S05804_W000003",
  "audio_path": "examples/mini/audio/EN_B00004_S05804_W000003.mp3",
  "align_word": [["Books", 0.0, 0.32], "..."],
  "original_text": "Books about the law are notorious for being very long, very dull, and very difficult to read.",
  "edited_text": "Books about the law are notorious for being very long, dull, and very difficult to read.",
  "instruction": "Please delete the second 'very'.",
  "reverse_instruction": null,
  "edits": [{"op": "deletion", "target": "very", "new": "", "direction": "", "region": [55, 59]}],
  "num_edits": 1,
  "meta": {"locator": "nth_occurrence", "target": "very", "count": 3, "occurrences": [[44, 48], [55, 59], [70, 74]], "ordinal": 2}
}
```

### 7.3 Final Exported Pair (`pairs.jsonl`)

```json
{
  "id": "single_span/insertion/ZH_B00011_S05354_W000006",
  "task": "single_span",
  "language": "zh",
  "edit_type": "insertion",
  "original_audio": "examples/mini/audio/ZH_B00011_S05354_W000006.mp3",
  "synthetic_audio": "outputs/mini/audio/single_span/insertion/ZH_B00011_S05354_W000006.wav",
  "original_text": "皇上对太子已经产生了猜忌之心，太子的日子好过不了了。",
  "synthetic_text": "皇上对太子已经产生了一丝猜忌之心，太子的日子好过不了了。",
  "original_duration": 7.233,
  "synthetic_duration": 7.84,
  "forward_instruction": "在‘产生了’后面加上‘一丝’",
  "reverse_instruction": "删除‘一丝’",
  "edit_regions": [{"original_start": 2.4, "original_end": 2.4, "synthetic_start": 2.4, "synthetic_end": 3.0}],
  "num_edits": 1,
  "extra": {
    "subset": "insertion",
    "edits": [{"op": "insertion", "target": "产生了", "new": "一丝", "direction": "after", "region": [9, 10]}],
    "meta": {},
    "align_word": "...",
    "asr_text": "皇上对太子已经产生了一丝猜忌之心，太子的日子好过不了了。",
    "asr_score": 0.0,
    "sim": 0.9303
  }
}
```

---

## 8. Differences from the Released VoxCanvas-41M Dataset

This repository is a clean, self-contained re-implementation of our internal production pipeline. While preserving all core algorithms, prompts, and templates, it introduces several bug fixes, stricter data gates, and environment simplifications:

### 8.1 Pipeline & Packaging Differences

| Component | Internal Production Pipeline (VoxCanvas-41M Release) | Open-Source Pipeline (This Repo) |
| --- | --- | --- |
| **Post-Export Selection** | Top-K ranking by UTMOS (`single_span`, top 5M/lang/op) or SIM (`multi_span` stratified by duration, `location_reference` top 50k/subset/lang), plus benchmark utterance exclusion. | Exports all pairs passing the paper's three baseline filters (`dur <= 30s`, `asr == 0`, `sim > 0.5`). |
| **Output Format** | Hugging Face Dataset with embedded audio and pre-swapped ChatML `messages`. | Path-based `.jsonl` with unified `extra` dict schema (`{subset, edits, meta, align_word, asr_text, asr_score, sim}`). |
| **ASR Backend** | Dedicated `vLLM` servers (`align` and `verify.asr`). | In-process `transformers` backend (`bf16`, batch 32) for single-environment compatibility (~85% exact transcript match; 99.6% timestamp match on identical tokens). |
| **LLM Endpoint** | Self-hosted `Qwen3.6-35B-A3B` (`top_k=20`). | Configurable OpenAI-compatible API (defaults to `deepseek-chat`) with an explicit `LANGUAGE_HINT` system prompt. |
| **SIM Batching** | Length-bucketed, padded, masked batches. | Exact single-utterance evaluation (`batch_size=1`). |

### 8.2 Key Algorithmic Refinements & Bug Fixes

1. **Universal Alignment & Token-Boundary Gate (§4.1)**:
   - Production only gated `d4` and allowed edits that caused token-boundary shifts (e.g., orphan apostrophes in `en` contractions or merged number runs like `21起500` $\to$ `21500` in `zh`). This repo applies `text.gate` and `edit_keeps_alignment` across all 16 subsets prior to synthesis.
   - Synchronizes the synthesizer's diff tokenizer with the aligner's exact granularity (`text.tokenize_aligned`), preventing unnecessary re-synthesis of hyphenated words or abbreviations (`U.S.`, `5:00`).

2. **Whole-Word & Case-Insensitive Uniqueness (§4.1, §4.6)**:
   - Production used case-sensitive substring matching (`str.count`) in several modules, causing false substring collisions (e.g., `"is"` inside `"compromised"`) or missing case-variant duplicates (`"The"` vs `"the"`). All uniqueness, adjacency (`is_valid_insertion`), and reversibility (`undo_insertion`) checks are now strictly case-insensitive and word-bounded.
   - Chinese deletion reverse contexts now grow by whole words (`jieba`) rather than single characters.

3. **Location-Reference Consistency (§4.5, §4.7)**:
   - **Timestamp Disambiguation (`l_t_*`)**: Drops candidates where 1-decimal rounded timestamps fail to uniquely resolve back to the edited words (~3.8% ZH, ~10.2% EN).
   - **Surface Visibility (`d1, m1, i1`)**: Enforces that the surface string appears exactly `count` times, preventing unedited leftovers when `jieba` merges a repeat into a compound word. Caps `m2/i2` ordinals at 8 (matching `d4`).

4. **Explicit Synthesis Guardrails (§5)**:
   - Explicitly intercepts and logs `no_context` (when the $\pm 0.5\text{ s}$ mask covers the entire utterance, which previously caused unconditional synthesis in a random voice, ~0.48% of rows), `too_long` ($> 8192$ frames), and `no_change`.
   - Enforces deterministic per-record RNG seeds during CFM sampling and normalizes Vocos waveform overshoots ($> 1.0$) before saving 16-bit PCM WAVs.
