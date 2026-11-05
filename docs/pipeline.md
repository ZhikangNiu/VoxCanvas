# VoxCanvas content-editing pipeline

This document describes how this repository builds instruction-based speech-editing pairs from real speech: the
content-editing part of VoxCanvas-41M (single-span, multi-span and location-reference editing on the Chinese and
English subsets of Emilia). For each stage it gives the algorithm, the parameters and the data format, and it lists every
known difference from the released VoxCanvas-41M data.

## 1. Overview

```
manifest.jsonl            {uuid, audio_path, language}
    |
    | 0  voxcanvas.align           Qwen3-ASR-1.7B transcript + Qwen3-ForcedAligner-0.6B word times       GPU
    v
aligned.jsonl             + text, align_word
    |
    | 1  voxcanvas.annotate        gate -> rule-based edit selection -> LLM new text -> templates         CPU + LLM API
    |                              (one run per task/subset; outputs are concatenated)
    v
edits.jsonl               edit records: original/edited text, forward/reverse instruction, edits
    |
    | 2  voxcanvas.synthesize      F5-TTS masked infilling of the edited regions + Vocos                  GPU
    v
synth.jsonl + wavs        + status, synthetic_audio, durations, edit_regions
    |
    | 3a voxcanvas.verify.asr      Qwen3-ASR-1.7B on the synthetic audio; en WER, zh tone-pinyin PER     GPU
    v
asr.jsonl                 + asr_text, asr_score
    |
    | 3b voxcanvas.verify.sim      WavLM-Large ECAPA-TDNN speaker similarity, original vs synthetic       GPU
    v
sim.jsonl                 + sim
    |
    | 4  voxcanvas.export          keep: status ok, duration <= 30 s, asr_score == 0, sim > 0.5           CPU
    v
pairs.jsonl               final pairs (release schema, audio as paths)
```

Every stage is a CLI (`python -m voxcanvas.<stage> --help`) that reads one jsonl file and writes another. All stages
except `export` append to their output and skip ids that are already there, so an interrupted run resumes where it
stopped; a partial last line left by a killed run is cut off, with a warning, before appending. The GPU stages (`align`,
`synthesize`, `verify.asr`, `verify.sim`) accept `--shard-id/--num-shards` (they process the rows whose
`zlib.crc32(id)` (align: `uuid`) `% num_shards == shard_id`, so a row keeps its shard when the input grows or is
reordered between runs) and use the visible GPU. A different `--num-shards` re-partitions the rows while each shard
still resumes from its own output file, so keep it fixed for a given stage and output, or delete the old shard files
first; otherwise done rows are computed again and the merged file can hold an id twice (for `align`, `annotate` then
rejects the repeated uuid, so re-run align into fresh shard files; for later stages `export` drops the second copy as
`duplicate`). `scripts/run_sharded.sh` starts one process per GPU (shard i on the i-th entry of
`CUDA_VISIBLE_DEVICES`, or GPU i when it is unset; it exits before starting any shard if the variable lists fewer than
`<num_gpus>` GPUs, e.g. none when it is set but empty), waits for all of them and concatenates the shard outputs in
shard order (so the merged file is grouped by shard, not in input order); if any shard fails it merges nothing and
exits non-zero, and an interrupt stops all shards. Every finished shard leaves an output file (empty for a shard
without rows), and the merge goes through a temporary file, so a failed merge leaves the previous merged output
untouched.

| stage | paper | code |
|---|---|---|
| align | §3.2 | `src/voxcanvas/align.py` |
| annotate | §3.3, App. A | `src/voxcanvas/annotate/` (`content.py`, `occurrence.py`, `boundary.py`, `timestamp.py`; shared `templates.py`, `prompts.py`, `llm.py`, `records.py`, `../text.py`) |
| synthesize | §3.4 | `src/voxcanvas/synthesize.py` |
| verify | §3.5 | `src/voxcanvas/verify/asr.py`, `src/voxcanvas/verify/sim.py` |
| export | §3.5 | `src/voxcanvas/export.py` |

### 1.1 Checkpoints

The stages read their checkpoints from `checkpoints/` by default (relative to the repository root, the working
directory of every command). `scripts/download_checkpoints.sh [hf|modelscope]` downloads every checkpoint except the
WavLM one, and a re-run skips the complete files. Both sources give byte-identical weights and configs (sha256 below);
only the repos' `.gitattributes` differ, and ModelScope adds a `configuration.json` that nothing reads.

| path in `checkpoints/` | size | `hf` (default) | `modelscope` |
|---|---|---|---|
| `F5TTS_v1_Base/model_1250000.safetensors` | 1.35 GB | `SWivid/F5-TTS` | `SWivid/F5-TTS_Emilia-ZH-EN` (the F5-TTS authors' ModelScope repo, file at its root) |
| `vocos-mel-24khz/` (`config.yaml`, `pytorch_model.bin`) | 54 MB | `charactr/vocos-mel-24khz` | the same repo from hf-mirror.com (not on ModelScope) |
| `Qwen3-ASR-1.7B/` (whole repo) | 4.7 GB | `Qwen/Qwen3-ASR-1.7B` | `Qwen/Qwen3-ASR-1.7B` |
| `Qwen3-ForcedAligner-0.6B/` (whole repo) | 1.8 GB | `Qwen/Qwen3-ForcedAligner-0.6B` | `Qwen/Qwen3-ForcedAligner-0.6B` |
| `wavlm_large_finetune.pth` | 1.3 GB | manual download from [Google Drive](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view), as in the F5-TTS evaluation | same |

- `hf` mode runs the `hf` CLI of the environment, which honours `HF_ENDPOINT` and keeps its download state in
  `.cache/huggingface/` inside each download directory. `modelscope` mode runs the ModelScope CLI as
  `uvx --from modelscope-hub==0.6.1 modelscope` (uv fetches it into its own cache on first use; it is not a project
  dependency) and the `hf` CLI with `HF_ENDPOINT=https://hf-mirror.com` for Vocos, with
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` so a stored Hugging Face token is not sent to the mirror.
  `pengzhendong/vocos-mel-24khz` on ModelScope is a byte-identical community copy of the Vocos files.
- `synthesize` uses the `vocab.txt` bundled with `f5_tts` (the vocabulary of the checkpoint repos), so no vocab file
  is downloaded.
- Other checkpoints are passed with `--ckpt` / `--vocoder` (synthesize), `--asr-model` / `--aligner-model` (align),
  `--model` (verify.asr) and `--ckpt` (verify.sim). A path under `checkpoints/` that does not exist fails before any
  model is loaded, naming the download script; verify.sim fails the same way for any missing `--ckpt`, naming the
  Google Drive link. Other values go to the loaders unchanged, so Hugging Face ids (`--asr-model`, `--aligner-model`,
  `--model`, `--vocoder charactr/vocos-mel-24khz`) and `hf://` URLs (synthesize `--ckpt`) are still downloaded at run
  time.
- verify.sim also needs s3prl's WavLM-Large code and `wavlm_large.pt`, which `f5_tts.eval.ecapa_tdnn` fetches with
  `torch.hub` (§6).

To check the downloads, run `sha256sum -c` in `checkpoints/` on:

```
670900fd14e6c458b95da6e9ed317cdb20dbaf7a1c02ac06a05475a9d32b6a38  F5TTS_v1_Base/model_1250000.safetensors
da9033922f969a47f0c160010226919e59f27761fd5066f3828d46de6650b0fc  vocos-mel-24khz/config.yaml
97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a  vocos-mel-24khz/pytorch_model.bin
a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6  Qwen3-ASR-1.7B/model-00001-of-00002.safetensors
6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc  Qwen3-ASR-1.7B/model-00002-of-00002.safetensors
47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7  Qwen3-ForcedAligner-0.6B/model.safetensors
51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b  wavlm_large_finetune.pth
```

### 1.2 Throughput

Throughput and peak GPU memory per GPU process, measured on a shared H20 at the default settings:

| stage | throughput | peak GPU memory |
|---|---|---|
| align | about 7.5 clips/s (batch 32; about 1,400 GPU-hours for all 38.1M clips) | about 15 GiB |
| synthesize | about 1.5-1.8 edits/s (NFE 32) | about 2.5 GiB |
| verify.asr | about 7-9 edits/s (batch 32) | about 11 GiB |
| verify.sim | about 10 edits/s | about 3 GiB |

Throughput excludes model loading. Memory includes the CUDA context and is the larger of a run on typical clips (256
Emilia clips with a mean of 9.5 s for align, the 287 edits of the mini run for the other stages) and a run on 64
clips of 30 s, the Emilia maximum. On a smaller GPU, lower `--batch-size` of `align` and `verify.asr`.

`align` and `verify.asr` use the transformers backend of qwen-asr; the release used its vLLM backend, which is several
times faster but needs a separate environment (vLLM pins a different torch than F5-TTS).

## 2. Subsets and paper Table 1

| Table 1 row | `task` | `subset` | forward edit | LLM | trained direction |
|---|---|---|---|---|---|
| Single-span insertion (reverse of a deletion) | `single_span` | `deletion` | delete a unique span of 1-6 words | no | reversed |
| Single-span deletion (reverse of an insertion) | `single_span` | `insertion` | insert new words next to a unique anchor | yes | reversed |
| Single-span substitution | `single_span` | `substitution` | replace a unique span of 1-6 words | yes | reversed |
| Multi-span (2-6 edits) | `multi_span` | `multi` | 2-6 separated deletions / substitutions / insertions | yes | reversed |
| Location: ordinal | `location_reference` | `d4` `m2` `i2` | delete / replace / insert at the i-th occurrence of a repeated word or phrase | m2, i2 | forward |
| Location: all occurrences | `location_reference` | `d1` `m1` `i1` | delete / replace / insert at every occurrence | m1, i1 | forward |
| Location: sentence boundary | `location_reference` | `l_d` `l_m` `l_i` | delete 1-6 words / replace / insert at the sentence head or tail | l_m, l_i | forward |
| Location: time | `location_reference` | `l_t_d` `l_t_m` `l_t_i` | delete / replace / insert at a time point or interval | l_t_m, l_t_i | forward |
| Paralinguistic insertion / deletion | - | - | not in this repository | - | - |

Each pair has a forward instruction (original audio -> synthetic audio) and, for single-span and multi-span, a reverse
instruction (synthetic audio -> original audio). In the release, single-span and multi-span pairs are trained
**reversed** (paper §3.4: source = synthetic speech, target = real speech, instruction = reverse instruction), and
location-reference pairs are trained forward. Paper Table 1 shows the instructions in the trained direction, so e.g.
"Add 'all' after 'you'." is the reverse of a deletion and "Delete 'very' before 'good'." the reverse of an insertion.
Released sizes: about 10M pairs per single-span operation, 10M multi-span pairs, and 50k pairs per location subset and
language (1.2M in total; Table 1 lists 100K per location row).

## 3. Stage 0: align

`python -m voxcanvas.align --input manifest.jsonl --output aligned.jsonl [--asr-model checkpoints/Qwen3-ASR-1.7B]
[--aligner-model checkpoints/Qwen3-ForcedAligner-0.6B] [--batch-size 32] [--max-new-tokens 1024]`

- Every clip is re-transcribed with Qwen3-ASR-1.7B and force-aligned with Qwen3-ForcedAligner-0.6B (qwen-asr transformers
  backend, bf16): `transcribe(audio=paths, language=["Chinese" | "English"], return_time_stamps=True)`. The language is
  forced from the manifest (`zh` -> Chinese, `en` -> English); no language identification is run.
- `text` = the ASR transcript, stripped. Edits are made on this text, not on the Emilia transcript, because the aligner
  times align the ASR text (strict token agreement with the original Emilia text is only 27% en / 63% zh).
- `align_word` = `[[token, start_s, end_s], ...]` as returned by the aligner: one entry per CJK character (zh) or per
  word (en), on an 80 ms grid, original case.
- If a batch fails (CUDA OOM, missing file, undecodable audio), the clips are retried one by one. Clips that still fail,
  and clips with an empty transcript or no time stamps, are skipped and logged.

## 4. Stage 1: annotate

`python -m voxcanvas.annotate --task {single_span,multi_span,location_reference} --subset S --input aligned.jsonl
--output edits.jsonl [--num-samples 0] [--seed 0] [--concurrency 16] [--base-url https://api.deepseek.com]
[--model deepseek-chat] [--api-key-env DEEPSEEK_API_KEY] [--extra-body JSON]`

### 4.1 Common steps

1. **Gate** (`text.gate`). A row is used only if its text contains no `[UNK]` and no script its language label rules out
   (en: CJK, kana or hangul; zh: kana or hangul), is equal to the `align_word` tokens when tokenized at aligner
   granularity, and has at least 4 content tokens. Emilia labels some Japanese, Korean and code-switched utterances zh
   or en; since every CJK character is its own aligner token, such rows would otherwise align, clear the content-token
   floor and get instructions in the wrong language. Kana punctuation is not matched, so the `・` (U+30FB) of
   transliterated names does not reject a zh row. Aligner granularity (`text.tokenize_aligned`) follows
   Qwen3-ForcedAligner: each CJK character is one token, and so is each run of other non-space characters, keeping only
   letters, digits and `'` (`5:00` -> `500`, `peer-reviewed` -> `peerreviewed`, `U.S.` -> `us`); `’` is dropped like
   other punctuation (`it’s` -> `its`). zh also splits Latin letters, so zh rows with English words fail the gate on
   purpose: zh edits put no spaces around words, and zh shards contain English utterances. The gate also rejects rows
   where one aligned token covers two content tokens (`U.S.`, `5:00`, `1,000`, `a.m.`), so every content token has its
   own time span. After the gate, the k-th aligned token of the text is `align_word[k]`, which the time-based subsets
   and the synthesizer rely on. Measured on the first 20,000 rows of B00000 / B00001 / B00060: en passes 95.9% / 94.8% /
   93.8% (no token mismatches; 4.0% / 5.0% / 6.1% rejected by the shared-token check); zh passes 95.7% / 96.2% / 82.1%
   (4.2% / 3.8% / 17.8% have English words). The foreign-script check rejects at most 0.1% of these rows, more where a
   shard holds a block of another language (162 of the first 30,000 rows of EN_B00001).
2. **Candidate** (`make_candidate`, rule-based, no LLM): at most one candidate per row and subset, with a length bucket.
3. **Sampling** (`records.stratified_sample`, only with `--num-samples N > 0`): candidates are grouped by length bucket,
   each bucket is ordered by a random priority per candidate, and buckets are visited round-robin starting from the
   longest until N candidates are picked; the picks keep input order. `--num-samples` counts candidates, so LLM subsets
   can yield fewer than N records. The picks depend on the set of candidates, not on their order, but resuming with
   `--num-samples` assumes the same input rows: after the aligned input grows, new rows can outrank earlier picks, so
   annotate into a fresh output (resuming would keep the old records and add the new picks).
4. **Completion** (`complete_candidate`, in a thread pool of `--concurrency` workers): LLM text, edit application,
   checks and templates, giving an edit record or nothing. Records are written in input order. Dropped candidates are not
   recorded, so re-running the stage on the same output retries them (same random streams, new LLM answers). The resume
   key `{task}/{subset}/{uuid}` does not include `--seed`, so a changed seed only takes effect in a fresh `--output`.

Shared rules:

- **Content tokens** (`text.tokenize_content`): zh = jieba words, except that embedded Latin/digit runs are split with
  the English word regex (`It's` stays one token); en = `[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*`. Punctuation is never a
  token. A punctuation segment starts wherever the gap between two tokens contains punctuation.
- **Uniqueness** (`text.count_occurrences`): case-insensitive and overlapping; in en a match must not extend a word on
  either side. Every "unique" in this document means a count of exactly 1.
- **Edit application** (`text.apply_edits`): non-overlapping character-span edits, applied right to left (never
  `str.replace`). In en a deletion also removes one adjacent space and an insertion is separated by a space. Unless
  all edits are substitutions, en double spaces and spaces before `, . ! ? ; :` are removed and the text is stripped.
  If any edit is a deletion, runs of punctuation collapse to their last mark, and if a deletion starts before any
  content, leading punctuation is stripped.
- **Content key** (`text.content_key`): the lowercased text restricted to `[0-9a-z]` and CJK. Two texts with the same
  key have the same spoken content; reversibility checks compare content keys.
- **Alignment check** (`text.edit_keeps_alignment`, in `records.build_record`): a candidate is dropped when its edits
  change the aligner tokenization outside the edited regions, i.e. unless the aligned tokens of the edited text are
  those of the original with each edit applied token by token. Deleting 起 from zh `21起500` glues `21500` into one
  token, and deleting `he` from en `thinking he'` leaves `thinking'`; the synthesizer diffs aligned tokens (§5), so it
  would also regenerate words the instruction does not name. On the first 20,000 gated rows per language of B00001,
  this drops 41 of 180,750 records of the five subsets without an LLM (0.023%: 2 zh digit merges, 39 en orphan
  apostrophes).
- **Random streams**: `rng_for(seed, uuid, subset, "candidate")` for selection and
  `rng_for(seed, uuid, subset, "complete")` for everything drawn during completion (length targets, template choice,
  reverse-context side, shuffles); `rng_for(seed, uuid, subset, "sample")` for the sampling priority of a candidate.
  Each stream is `random.Random("-".join(keys))` and keyed by the row, so results do not depend on the input order or
  on sharding (only on the LLM); this includes which candidates `--num-samples` picks.

### 4.2 LLM generation

- Client (`annotate/llm.py`): any OpenAI-compatible chat endpoint (`--base-url`, `--model`, key from the environment
  variable named by `--api-key-env`, also read from `.env`). temperature 1.0, top_p 0.95, max_tokens 256, timeout 120 s,
  3 SDK retries. `--extra-body` is passed as `extra_body`, e.g.
  `{"top_k": 20, "chat_template_kwargs": {"enable_thinking": false}}` for a vLLM / SGLang Qwen3 server (`top_k` 20 as in
  the release, §9.4). Each request has a system message `LANGUAGE_HINT[lang]` ("The input sentence is English, so
  answer in English.") and the prompt as the user message.
- Prompts (`src/voxcanvas/annotate/prompts.py`, verbatim from the production pipeline; paper Appendix A prints
  shortened versions):

  | prompt | used by | placeholders |
  |---|---|---|
  | `SUBSTITUTION_PROMPT` | single_span substitution, multi_span substitutions | `{raw_text} {target} {length_hint}` |
  | `INSERTION_PROMPT` | single_span insertion, multi_span insertions | `{raw_text} {target} {length_hint} {direction} {direction_upper} {direction_zh} {splice_order}` |
  | `GLOBAL_SUBSTITUTION_PROMPT` | m1 m2 l_m l_t_m | `{raw_text} {target}` |
  | `LOCATION_INSERTION_PROMPT` | i1 i2 l_i l_t_i | `{raw_text} {target} {direction} {direction_zh}` |

  `{length_hint}` (`prompts.length_hint`) asks for "roughly N word(s)" (en) or "N Chinese character(s)" (zh).
  `INSERTION_PROMPT` rendered for `before` / `after` is byte-identical to the production insert-before / insert-after
  prompts.
- Answer: `<think>...</think>` blocks are removed; the answer is the last non-empty line, with a leading label
  (`Answer:`, `输出：`, ...) and edge quotes removed. Up to 5 draws; a draw is accepted only if it passes the validator
  and the subset's `accept` check, otherwise the next draw is requested. A failed request (after SDK retries) counts as a
  failed draw; authentication / permission / not-found errors stop the run. If all 5 draws fail, the candidate is dropped.
  If not a single request of the run gets a reply (e.g. a misconfigured `--base-url` / `--model` / `--extra-body`
  rejected with HTTP 400 every time), the stage logs its summary and then exits non-zero with a `RuntimeError` instead
  of finishing successfully; only an empty output file is left behind, which a re-run against a working endpoint fills.
- Validators (`text.is_valid_replacement`, `text.is_valid_insertion`): non-empty, contains a letter / digit / CJK
  character, no punctuation, symbol, space or control character (Unicode P/S/Z/C) at either edge, same language (en:
  no CJK; zh: must contain CJK unless the target has none). Replacement: not already in the sentence
  (case-insensitive substring, so never the target itself) and not a pure homophone swap within 他她它, 的地得, 在再,
  做作, 座坐, 那哪, 了啦, 吗嘛, 呢呐, 和合. Insertion: must not contain the anchor and must not already stand next to
  the anchor on that side (with or without a space, any case).
- Edit-level `accept` checks of every LLM subset: the edit changes the text, it creates no new adjacent duplicate
  (`text.has_adjacent_duplicate`: zh 1-4 CJK characters repeated immediately, en the same word twice in a row; ignored if
  the sentence already had one), and it is reversible: replacing every occurrence of the new text by the target (or
  deleting the inserted text next to the anchor everywhere, `text.undo_insertion`, case-insensitive) restores the
  original content key.

### 4.3 single_span (`content.py`)

- A token index `idx` is drawn uniformly over the content tokens.
- **deletion**: k ~ U{1..min(6, n - idx, n - 2)} tokens from `idx`. **substitution**: k ~ U{1..min(6, n - idx)}.
  The span grows one token at a time, alternately left and right, until it is unique in the sentence; it may cross
  punctuation. If the unique span is longer than 60% of the sentence characters, it is retried with k - 1 tokens. No
  unique span -> no candidate.
- **insertion**: direction ~ U{before, after}. The anchor starts as token `idx` and grows away from the insertion point
  (left for "after", right for "before"), inside its punctuation segment, until unique. No unique anchor -> no candidate.
- LLM length targets: substitution `max(1, round(len(target) * U(0.5, 3)))`, insertion U{1..20}, in words (en) or
  content characters (zh). The answer may not exceed the target length by more than 20 units.
- Forward templates: `删掉‘X’` / `Delete 'X'.`; `把‘X’改成‘Y’` / `Replace 'X' with 'Y'.`; `在‘X’前面加上‘Y’`,
  `在‘X’后面加上‘Y’` / `Add 'Y' before 'X'.`, `Add 'Y' after 'X'.`.
- Reverse instructions:
  - substitution: X and Y swapped;
  - insertion (partial anchor): `删除‘Y’` / `Delete the 'Y'.` if Y is unique in the edited text, else
    `删除‘X’前面的‘Y’` / `Delete the 'Y' before 'X'.` (or 后面 / after) if Y with its anchor is unique, else the
    draw is rejected and redrawn;
  - deletion: the deleted text is added back before or after a context of whole words next to the deletion point
    (`templates.deletion_reverse`): the two sides are tried in random order; on each side the context grows word by word
    away from the deletion point until it is unique in the edited text, and it must restore the original content key.
    In single_span, if the first unique context on a side fails that content check, the side is abandoned. Contexts
    never start or end with punctuation.
- `length_bucket` = number of content tokens of the sentence: `4-7`, `8-15`, `16+`. `meta` = `{}`.

### 4.4 multi_span (`content.py`)

- Target number of edits k ~ U{2..6}. Up to 8k placement tries: a random unused token, an edit kind drawn with weights
  deletion : substitution : insert-before : insert-after = 2 : 2 : 1 : 1, then the same span / anchor selection as
  single_span. Each placed edit also reserves one token on each side, so two edits are at least two tokens apart. Fewer
  than 2 placed edits -> no candidate.
- One LLM call per substitution / insertion region (same prompts and length targets as single_span; only the length
  delta is checked per region). Afterwards the whole edit must change the content, must not create a new adjacent
  duplicate, and the reverse chain (reverse sub-edits replayed right to left) must restore the original content key.
- Forward instruction: sub-instructions in text order, joined with `INSTRUCTION_SEP` = `" "` (en) / `"，"` (zh).
  Reverse instruction: the sub-reverses (as in single_span, deletion contexts without the per-edit content check) in a
  shuffled order, joined the same way; insertion reverses use the bare form drawn from `删除‘Y’` / `删掉‘Y’` or
  `Delete 'Y'.` / `Delete the 'Y'.` when Y is unique, the anchored form otherwise, and the sample is dropped if even
  the anchored form is ambiguous. A substitution reverse names its new text alone, so the sample is also dropped when
  a substitution's new text does not occur exactly once in the edited text (it can collide with another sub-edit's new
  text or with text already present). All reverse sub-instructions refer to the edited text as a whole (they are valid
  applied together, or right to left).
- `num_edits` = number of placed edits; `meta` = `{"num_edits_target": k}`; length bucket as single_span.

### 4.5 Occurrence subsets d1 d4 m1 m2 i1 i2 (`occurrence.py`)

- `find_repeats`: maximal repeated token n-grams, never crossing punctuation. Every n-gram of lowercased content tokens
  inside a punctuation segment is collected; an n-gram that occurs at least twice is kept unless all its occurrences
  have the same neighbour token on the right, or all on the left (then a longer repeat contains it).
- The repeat with the most tokens is chosen (ties: earliest first occurrence) among the eligible ones: occurrences do
  not overlap and the surface text is visible exactly `count` times (`count_occurrences`), so "all" / the ordinal
  refers to every visible match; for d4/m2/i2 at most 8 occurrences; for d1, deleting every occurrence leaves some
  content.
- d4/m2/i2 edit occurrence i ~ U{1..count}; d1/m1/i1 edit every occurrence. Insertion side ~ U{before, after}.
- The target shown in the instruction and sent to the LLM is the surface text of the edited occurrence(s) (lowercased
  first surface if the occurrences differ in case). m1/m2 use `GLOBAL_SUBSTITUTION_PROMPT`, i1/i2
  `LOCATION_INSERTION_PROMPT` (no length hint).
- Templates (`templates.D1 D4 M1 M2 I1 I2`), one drawn at random, e.g. `删除第二个‘的’`, `Delete the second 'the'.`,
  `把所有的‘红色’改成‘蓝色’`, `Replace all instances of "red" with "blue".`, `Insert "still" after each "was".`.
  Ordinals: 一..八 (zh), first..eighth (en) (at most 8 occurrences).
- No reverse instruction. `num_edits` = count for d1/m1/i1, 1 for d4/m2/i2. `meta` = `{"locator": "all_occurrences" |
  "nth_occurrence", "target", "count", "occurrences": [[s, e], ...], "ordinal"}` (ordinal for d4/m2/i2 only).
  Length bucket by repeat length in tokens: `1`..`5`, `6-7`, `8-10`, `11-15`, `16+`.

### 4.6 Sentence-boundary subsets l_d l_m l_i (`boundary.py`)

- Head / tail are the start and end of the whole transcript: there is no sentence splitting, so a multi-sentence clip
  has a single head and a single tail, and "sentence-final" then means the last words of the utterance (as in the
  release).
- Position ~ U{head, tail}. l_d deletes k ~ U{1..min(r, 6, n - 2)} edge tokens, where r is the number of tokens in the
  edge punctuation segment (at least two tokens are kept). l_m replaces the single edge token
  (`GLOBAL_SUBSTITUTION_PROMPT`); l_i inserts before or after it (`LOCATION_INSERTION_PROMPT`).
- Templates `L_D`, `L_M`, `L_I`, e.g. `删除句首的‘所以’`, `Delete the sentence-final "do a tie".`,
  `Insert "Honestly" before the first word "Well".`. The en templates "the first word" / "the last word" are also used
  when l_d deletes several words (as in the release).
- No reverse instruction. `meta` = `{"locator": "sentence_position", "position": "head" | "tail", "num_tokens": k}`.
  Length bucket: l_d by k (`1`..`5`, `6+`), l_m / l_i by sentence length (`4-7`, `8-15`, `16+`).

### 4.7 Time subsets l_t_d l_t_m l_t_i (`timestamp.py`)

- Word times: each content token spans from the start of the first to the end of the last `align_word` entry it
  overlaps. Times are relative to the start of the original (untrimmed) audio.
- Mode ~ U{point, span} (l_t_i: point only). point: one random word, located by its midpoint t. span: 1-5 consecutive
  words inside one punctuation segment, located by [start of the first, end of the last]. Times are shown with one
  decimal (`f"{t:.1f}"`).
- Disambiguation: the candidate is dropped unless the shown time(s) pick out exactly the edited words. point: the words
  whose span contains the shown time, or else the nearest word(s); span: the words whose midpoint lies in the shown
  interval. This drops about 4% (zh) / 10% (en) of rows. l_t_d also keeps at least two tokens.
- has_word ~ Bernoulli(0.5): with the word (`删除第0.7秒的‘风口’`, `Remove "dinner with Ivanka" between 4.6s and 6.1s.`)
  or with the time only (`删除第3.0到5.0秒的内容`, `Delete the content at 2.8s.`). Templates `L_T_D`, `L_T_M`, `L_T_I`.
- No reverse instruction. `meta` = `{"locator": "timestamp", "mode", "has_word", "t_start", "t_end", "token_range":
  [lo, hi]}`. Length bucket: l_t_d by deleted words (`1`..`4`, `5+`), l_t_m / l_t_i by sentence length.

## 5. Stage 2: synthesize

`python -m voxcanvas.synthesize --input edits.jsonl --output synth.jsonl --audio-dir DIR
[--ckpt checkpoints/F5TTS_v1_Base/model_1250000.safetensors] [--vocoder checkpoints/vocos-mel-24khz] [--vocab ""]
[--nfe-step 32] [--cfg-strength 2.0] [--sway-sampling-coef -1.0] [--seed 0]`

The edited speech is generated by F5-TTS (F5TTS_v1_Base, upstream `f5_tts` 1.1.20 as a git submodule) by masked
infilling: the original mel spectrogram is kept outside the edited regions and regenerated inside them, conditioned on
the full edited transcript.

1. **Audio**: `torchaudio.load`, channel mean; if the RMS is below 0.1 the waveform is scaled up to 0.1; resampled to
   24 kHz; Vocos log-mel (100 bins, hop 256, 93.75 frames/s).
2. **Edit plan** (`build_edit_plan`): `difflib.SequenceMatcher(autojunk=False)` between the normalized `align_word`
   tokens and `text.tokenize_aligned(edited_text, language)` (the gate tokenizer). Seconds per token = (last end -
   first start) / number of tokens with positive duration, clamped to [0.08, 0.32] s (zh) / [0.10, 0.45] s (en), or
   0.2 s (zh) / 0.24 s (en) when no aligned token has a positive duration. Per non-equal opcode:
   - replace: source part [start of first, end of last replaced word], target duration
     `max(0.18, duration * n_new / n_old)`;
   - insert: source part = the pause between the neighbouring words (the utterance start or end at the edges), target
     `max(0.18, n_new * seconds_per_token)`;
   - delete: source part = the deleted words, target 0.1 s of generated filler.

   Overlapping or touching parts are merged (targets summed), then shifted onto the edited timeline.
3. **Condition**: each source part of the mel is replaced by zero frames of its target duration (at least one frame).
4. **Mask**: the frames to generate are the edited-timeline parts widened by 0.5 s on each side (paper §3.4).
5. **Sampling**: `CFM.sample(cond, text=convert_char_to_pinyin([edited_text]), duration=num_frames, steps=32,
   cfg_strength=2.0, sway_sampling_coef=-1.0, edit_mask=mask, seed=...)` with a per-record seed
   `rng_for(--seed, id).getrandbits(32)`; Vocos decodes the whole mel; the RMS boost is undone; if the peak exceeds 1
   the wav is scaled to peak 1 (no clipping); the wav is written as 24 kHz mono PCM_16 to `<audio-dir>/<id>.wav`, i.e.
   one directory per task and subset (e.g. `<audio-dir>/single_span/deletion/<uuid>.wav`). Outside the mask the mel
   equals the original, but the whole waveform is re-vocoded (nothing is pasted back).
6. **Status**: `no_change` (the diff finds no spoken change), `too_long` (the condition would exceed 8192 mel frames,
   about 87 s, the DiT position limit), `no_context` (the 0.5 s-widened mask covers every frame of the condition, so no
   original frame is left as the speaker prompt and sampling would be unconditional TTS in another voice), else `ok`.
   Only `ok` rows get a wav. A record whose audio file does not exist gets no row: a warning is logged and the run
   counts it as skipped (missing audio) in the summary line. Every other row is still written, then the run fails with
   `FileNotFoundError` (non-zero exit), so `run_sharded.sh` merges nothing. The id stays undone, so a re-run after
   fixing the paths retries only those records (align decoded every clip, so a missing file at synthesis time is an
   environment error, e.g. the wrong working directory or an unmounted dataset root). Any other OS error raises
   immediately. The run ends with
   `Wrote <num_written>/<num_records> rows in Xs (Y rows/s): {statuses}; <num_missing> skipped (missing audio)`,
   where `{statuses}` counts only the written rows (`ok` / `no_change` / `too_long` / `no_context`); it is logged
   before the `FileNotFoundError`.
7. `edit_regions`: for each merged part, `original_start/original_end` (source timeline) and
   `synthetic_start/synthetic_end` (edited timeline, without the 0.5 s widening), rounded to 4 decimals.

## 6. Stage 3: verify

**ASR** (`python -m voxcanvas.verify.asr --input synth.jsonl --output asr.jsonl [--model checkpoints/Qwen3-ASR-1.7B]
[--batch-size 32]`). Qwen3-ASR-1.7B (transformers backend, bf16) transcribes each synthetic wav without a language hint.
The transcript is scored against `edited_text` with jiwer (`asr_score`, 0 = perfect):

- en: lowercase, every character outside `[\w\s']` becomes a space, spaces collapse; word error rate.
- zh: whitespace and every Unicode punctuation (P\*) and symbol (S\*) character are removed; both texts are converted
  with `pypinyin.lazy_pinyin(style=TONE3, neutral_tone_with_five=True)` and compared as syllable sequences (a
  tone-sensitive pinyin error rate, so homophonous characters with the same tones are not penalized).

**SIM** (`python -m voxcanvas.verify.sim --input asr.jsonl --output sim.jsonl
[--ckpt checkpoints/wavlm_large_finetune.pth]`).
`f5_tts.eval.ecapa_tdnn.ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large")` with the UniSpeech WavLM-Large
speaker-verification checkpoint; both wavs are downmixed and resampled to 16 kHz; `sim` = cosine similarity of the
embeddings of the whole original and the whole synthetic utterance (batch size 1).
The ECAPA model loads s3prl's `wavlm_large` from `~/.cache/torch/hub/s3prl_s3prl_main` if present, else from GitHub with
`torch.hub` (which downloads `wavlm_large.pt`, 1.26 GB, from huggingface.co to `~/.cache/s3prl/download`); offline
machines need both caches.

Rows whose status is not `ok` pass through both stages unchanged.
If an ASR batch fails (CUDA OOM, missing file, undecodable audio) its wavs are retried one by one, as in align; in SIM
a row whose original or synthetic wav cannot be loaded fails on its own. Such an `ok` row gets no output row; every
other row is written, then the run fails with `RuntimeError` (non-zero exit) so `run_sharded.sh` merges nothing, and a
re-run retries only those rows.

## 7. Stage 4: export

`python -m voxcanvas.export --input sim.jsonl --output pairs.jsonl [--max-duration 30] [--max-asr-score 0] [--min-sim 0.5]`

A row is kept iff `status == "ok"`, `synthetic_duration <= 30`, `asr_score <= 0` (en WER 0 / zh PER 0) and `sim > 0.5`.
Dropped rows are counted by their first failing check (`duplicate` for a repeated id after the first row with that
id, `no_change` / `too_long` / `no_context`, `over_max_duration`, `asr_mismatch`, `low_sim`), and a table of total /
kept / dropped counts per (task, subset, language) is logged. The output is overwritten.

## 8. Data formats

Paths are stored as given; relative paths are relative to the working directory (run every stage from the repository
root).

**8.1 manifest** (input of align): `{"uuid": str, "audio_path": str, "language": "zh" | "en", ...passthrough}`.
`uuid` must be a unique, non-empty string without `/` (it is part of the record ids and the wav paths); `align` and
`annotate` raise `ValueError` otherwise.

**8.2 aligned row** (output of align): the manifest row plus `"text"` (ASR transcript) and
`"align_word": [[token, start_s, end_s], ...]`.

```json
{"uuid": "EN_B00004_S05804_W000003", "audio_path": "examples/mini/audio/EN_B00004_S05804_W000003.mp3", "language": "en",
 "text": "Books about the law are notorious for being very long, very dull, and very difficult to read.",
 "align_word": [["Books", 0.0, 0.32], ["about", 0.32, 0.64], ["the", 0.64, 0.64], "..."]}
```

**8.3 edit record** (output of annotate):

| field | meaning |
|---|---|
| `id` | `f"{task}/{subset}/{uuid}"`, unique and stable |
| `task`, `subset` | see §2 |
| `edit_type` | forward operation: `deletion` / `insertion` / `substitution` / `multi` |
| `language`, `uuid`, `audio_path`, `align_word` | from the aligned row |
| `original_text`, `edited_text` | transcript before / after the edit |
| `instruction` | forward instruction (original -> edited) |
| `reverse_instruction` | edited -> original; `null` for location_reference |
| `edits` | one `{"op", "target", "new", "direction", "region"}` per edit, sorted by `region` start. `op` is `deletion` / `insertion` / `substitution`; `target` is the deleted / replaced text or, for an insertion, the anchor; `new` is the inserted / replacement text (`""` for deletion); `direction` is `before` / `after` for insertions, else `""`; `region` = `[start_char, end_char]` in `original_text` (insertion: of the anchor token next to the insertion point) |
| `num_edits` | multi: number of edits; d1 / m1 / i1: number of occurrences; else 1 |
| `meta` | subset-specific, see §4 |

```json
{"id": "location_reference/d4/EN_B00004_S05804_W000003", "task": "location_reference", "subset": "d4",
 "edit_type": "deletion", "language": "en", "uuid": "EN_B00004_S05804_W000003",
 "audio_path": "examples/mini/audio/EN_B00004_S05804_W000003.mp3", "align_word": "...",
 "original_text": "Books about the law are notorious for being very long, very dull, and very difficult to read.",
 "edited_text": "Books about the law are notorious for being very long, dull, and very difficult to read.",
 "instruction": "Please delete the second 'very'.", "reverse_instruction": null,
 "edits": [{"op": "deletion", "target": "very", "new": "", "direction": "", "region": [55, 59]}], "num_edits": 1,
 "meta": {"locator": "nth_occurrence", "target": "very", "count": 3, "occurrences": [[44, 48], [55, 59], [70, 74]], "ordinal": 2}}
```

**8.4 synthesized row** (output of synthesize): the edit record plus `"status"`, `"synthetic_audio"` (wav path, `""`
unless ok), `"original_duration"` (s, source audio), `"synthetic_duration"` (s, 0 unless ok) and `"edit_regions"`:
`[{"original_start", "original_end", "synthetic_start", "synthetic_end"}]` (empty unless ok).

**8.5 verified rows**: verify.asr adds `"asr_text"` and `"asr_score"`, verify.sim adds `"sim"` (ok rows only).

**8.6 pair** (output of export), mirroring the released schema with audio paths instead of embedded audio (§9.1
"Packaging" lists the remaining schema differences):

```json
{"id": "single_span/insertion/ZH_B00011_S05354_W000006", "task": "single_span", "language": "zh", "edit_type": "insertion",
 "original_audio": "examples/mini/audio/ZH_B00011_S05354_W000006.mp3",
 "synthetic_audio": "outputs/mini/audio/single_span/insertion/ZH_B00011_S05354_W000006.wav",
 "original_text": "皇上对太子已经产生了猜忌之心，太子的日子好过不了了。",
 "synthetic_text": "皇上对太子已经产生了一丝猜忌之心，太子的日子好过不了了。",
 "original_duration": 7.233, "synthetic_duration": 7.84,
 "forward_instruction": "在‘产生了’后面加上‘一丝’", "reverse_instruction": "删除‘一丝’",
 "edit_regions": [{"original_start": 2.4, "original_end": 2.4, "synthetic_start": 2.4, "synthetic_end": 3.0}], "num_edits": 1,
 "extra": {"subset": "insertion",
           "edits": [{"op": "insertion", "target": "产生了", "new": "一丝", "direction": "after", "region": [9, 10]}],
           "meta": {}, "align_word": "...", "asr_text": "皇上对太子已经产生了一丝猜忌之心，太子的日子好过不了了。",
           "asr_score": 0.0, "sim": 0.9302895069122314}}
```

To build a training example in the released direction: for `single_span` and `multi_span` use `synthetic_audio` as the
source, `original_audio` as the target and `reverse_instruction`; for `location_reference` use `original_audio` ->
`synthetic_audio` with `forward_instruction`.

## 9. Differences from the released VoxCanvas-41M data

The released data were produced by an internal pipeline. This repository re-implements it with the same rules,
parameters, templates and prompts, except for the differences below (bug fixes, simplifications and backend changes).
Measured effects come from comparisons against the production code and data on Emilia shards B00000-B00011.

### 9.1 Steps that are not part of this repository

- **Release selection** (not in paper §3.5): single-span kept the top 5M pairs per language and operation by UTMOS of
  the synthetic wav; multi-span kept pairs with ASR 0, SIM > 0.5 and duration < 30 s, then the top 5M per language by
  SIM, stratified over 30 one-second duration buckets; location-reference kept the top 50k by SIM per subset and
  language, with durations in [0.3, 30] s. Here `export` applies only the paper's three filters and keeps everything
  that passes.
- **Benchmark exclusion**: utterances used by the evaluation benchmarks were removed from the release.
- **Packaging**: the release is ChatML (`messages` with instruction + source audio, target audio; `metainfo`) and a
  Hugging Face dataset with embedded audio; reversed pairs swap the audio there. Here pairs are jsonl with audio paths.
  The remaining schema differences from the release: `id` is `{task}/{subset}/{uuid}` instead of
  `{task}/{language}/{edit_type}/{key}` (where `key` is `{subset}_{task_id}` for the location subsets), `extra` is a
  JSON object instead of a JSON string, and the release's two always-empty columns
  `forward_augmented_instructions` / `reverse_augmented_instructions` are not written. The keys inside `extra` also
  differ: every task gets the same `{subset, edits, meta (holds the locator), align_word, asr_text, asr_score, sim}`
  instead of the release's per-task keys (single-span `fix_duration`; multi-span `edit_types` / `sub_edits` /
  `edited_asr_text`; location-reference `locator` / `align_word` / `edited_asr_text`), so code written against the
  release that reads `extra["locator"]`, `extra["sub_edits"]` or `extra["edited_asr_text"]` needs adapting.
- **Paralinguistic editing** (39 nonverbal event classes, Table 1 rows "Paralinguistic") and **multi-turn / history
  editing** are not included.
- Human spot checks.

### 9.2 Instruction format notes

- **Multi-span join**: sub-instructions are joined with a space (en) and `，` (zh), not `;` / `；` as printed in Table 1.
  The forward instruction lists them in text order; the reverse instruction (the trained one) is shuffled.
- **Partial anchor**: the reverse of an insertion is a bare deletion (`Delete the 'Y'.` / `删除‘Y’`) whenever the
  inserted text is unique in the edited text, and anchored (`Delete the 'Y' before 'X'.` / `删除‘X’前面的‘Y’`) only
  otherwise; in the release about 93% of insertion reverses are bare. Multi-span bare deletions are drawn from
  `Delete 'Y'.` / `Delete the 'Y'.` and `删除‘Y’` / `删掉‘Y’`. The zh forward deletion template is `删掉‘X’`.
- **Multi-span generation**: paper §3.3 describes the model as generating all k edits in a single pass. The release and
  this repository place the k regions by rule and call the LLM once per substitution / insertion region (deletions need
  no call); see §4.4.
- **Released reverse direction**: single-span and multi-span are trained reversed (synthetic -> real, with
  `reverse_instruction`); location-reference stays forward and has no reverse instruction. Table 1 prints the reversed
  forms, so e.g. its single-span "Insertion" row is the reverse of a deletion.
- Time templates say "content" / "内容" and print one decimal (`Delete the content from 3.0s to 5.0s.`), unlike Table 1's
  "speech from 3 s to 5 s". Location templates keep the production quote styles: en d1/d4 use single quotes, the other
  location subsets double quotes.
- The released zh single-span deletion reverses used character-level contexts; this repository uses word-level ones
  (see 9.4).

### 9.3 align

- In-process qwen-asr transformers backend instead of vLLM. Decoding is greedy in both, but about 15% of transcripts
  differ slightly (256 clips: 84.8% identical text, CER 0.33% zh, WER 1.7% en); on identical token sequences 99.6% of the
  time stamps are identical. fp32 does not get closer, so the gap comes from the backend.
- Generic manifest input (`uuid`, `audio_path`, `language`) instead of Emilia B-files plus a language flag; the forced
  language is the same.
- One stage instead of transcription + conversion: the output has `text` and `align_word` only (no duplicated ASR text,
  Emilia transcript or wav fields, `audio_path` instead of the internal path field). A `text` field in the manifest is
  overwritten by the ASR text.
- WER / PER against the Emilia transcript are not computed (they were informational only).
- Clips with an empty transcript or no time stamps are skipped inside align (the converter dropped them later; the final
  set is the same).
- Only OOM / OS / decode errors are caught; the per-clip retry happens outside the except block. Output is flushed per
  row; resume key is `uuid`; shards are chosen by crc32 of the uuid. Defaults: batch 32, max_new_tokens 1024.

### 9.4 annotate: shared text processing and LLM

- **LLM**: the release used Qwen3.6-35B-A3B (paper §3.3); the default here is `deepseek-chat`, so the generated words
  differ. Pass `--base-url` / `--model` / `--extra-body` for a self-hosted Qwen3.6-35B-A3B (README, "LLM endpoint") to
  stay closest to the release.
- **One edit primitive** (`apply_edits`) for all subsets, verified byte-identical to the production edit functions on
  10,000 deletions, 5,493 insertions and 5,493 substitutions, with these fixes: en deletions no longer leave a space
  before punctuation (d1: 477 of 2,371 en rows differ only by this) and swallow exactly one adjacent space; leading
  punctuation is stripped exactly when a deletion starts before any content (fixes the multi-span is_head bug and an
  orphan leading mark in zh d4; d1 and l_t_d no longer strip it when the deletion is not at the head); punctuation runs
  collapse only when there is a deletion (production multi-span always collapsed them); en space collapsing touches
  spaces only. Kept from the release: zh spaces are never tidied, so deleting a spaced Latin letter or digit
  (`参数 k 究竟`) leaves a double space or a space before punctuation (45 of 319,238 released zh single deletions on
  B00000); content keys ignore spaces, so the reversibility and verify checks are unaffected.
- **Edits that re-tokenize the text are dropped** (new, §4.1 alignment check). An aligned token keeps a leading or
  trailing `'` that no content token covers (`poppin'`, `'em`; 0.49% / 0.65% / 0.54% of gated en rows in B00000 /
  B00001 / B00060, 0% zh), so deleting or replacing such a word moves the mark onto its neighbour (`get it poppin',
  shout` -> `get it', shout`). Such records used to be exported and then rejected by verify as `asr_mismatch` (3 of
  29,061 records across the 16 subsets). In zh, deleting a character between two numbers glues them (`21起500` ->
  `21500`); verify cannot catch this, because its ASR reference is the already-glued `edited_text`. Both are now
  dropped at annotate.
- **Uniqueness counting** is case-insensitive, overlapping and whole-word in en everywhere. Production used
  case-sensitive, non-overlapping `str.count` (or `\b` matches for ordinals), so e.g. "is" inside "compromised" forced
  span growth and "The" / "the" counted as distinct.
- **content_key** lowercases before filtering, so zh text keeps its Latin letters.
- **`[UNK]` rows are dropped by the gate**. Production mapped `[UNK]` to `unk` for the alignment comparison, but
  content tokenization splits the brackets, so an edit could cut the marker in half. 0 of 538,310 production aligned
  rows contain it.
- **Edge characters**: the validators reject any Unicode P/S/Z/C character at either edge (production used a fixed
  list, so `**banana**`, `- banana` and `` `x` `` passed). 16 of 80,000 production single-span answers would be
  redrawn, mostly unbalanced 《/》, a trailing `-` / `·` or a block of code.
- **Index draws** use `rng.randrange(n)` instead of production's `int(random() * n)` in every subset (token and word
  indices, single_span / multi_span included); the distribution is the same but individual draws differ.
- **Deletion reverse contexts** are whole words in both languages (production zh used characters), never start or end
  with punctuation, and start from the exact deletion point. 10,000 / 10,000 deletions get a context (production
  9,997).
- **Insertion adjacency check** (`is_valid_insertion`) compares spaced and unspaced forms, case-insensitively and
  whole-word, in every subset. Production compared only the unspaced, case-sensitive form, which never matched English
  and let duplicates such as "the the" through (about 14% of en l_i / l_t_i rows).
- `is_valid_replacement` tests "already in the sentence" case-insensitively.
- **Language check** (new): answers in the wrong language are rejected (deepseek-chat answered English insertion prompts
  in Chinese about 80% of the time).
- **System message** (new): `LANGUAGE_HINT` is sent with every prompt (production sent the user prompt only; the user
  prompt is unchanged). It reduced Chinese answers to the en insert-before prompt from 19/24 to 0/24.
- `undo_insertion` is word-bounded in en (production also matched "that the" inside "that there"; 4 / 2,688
  verdicts) and case-insensitive in both languages, keeping the matched anchor surface (production zh used
  case-sensitive `str.replace`, so i1 on zh repeats with mixed-case Latin such as ChatGPT / chatgpt was never
  reversible).
- **Answer extraction**: the last non-empty line for all prompts (production took the first line for insertions,
  although every prompt asks for the answer on the final line); leading labels are removed; an unclosed `<think>` gives
  an empty answer (a failed draw). Replies such as `1. Paris`, one-line preambles ("Sure! Here is the replacement:
  Paris"), refusals and multi-option answers (`Paris / Berlin`) are taken as given (production behaved the same; none
  were seen in 40 deepseek-chat calls).
- **Client**: SDK retries (3) instead of a manual 4 x 3 s loop; no endpoint probing or round-robin over servers;
  presence_penalty left at the API default; proxy variables untouched; no `extra_body` unless given, so the release's
  `top_k` 20 is not sent (temperature 1.0, top_p 0.95 and max_tokens 256 match; its min_p 0.0 and repetition_penalty
  1.0 are no-ops). vLLM's default `--generation-config auto` already applies the model's own `top_k` 20, so passing
  `--extra-body '{"top_k": 20, ...}'` on a self-hosted Qwen makes it explicit and independent of server settings
  (e.g. `--generation-config vllm`); the DeepSeek API ignores it.
- **Redraws**: edit-level checks (no change, new adjacent duplicate, reversibility, length delta) run inside the draw
  loop, so a failure triggers a redraw as in production.
- **Curly apostrophe**: the aligner keeps only ASCII `'`, so `it’s` is aligned as `its` while the content token stays
  `it’s`; no released row has an inner `’`.
- **Sampling**: `stratified_sample` returns picks in input order, has no per-bucket quota cap or oversampling, and
  `--num-samples` counts candidates, not kept records. Records are written in input order (production: completion
  order). Production shuffled each bucket with one `Random(seed)`, while here each candidate has its own priority
  stream, so the picks do not depend on the input order; the distribution is unchanged but individual picks differ.
- **Random streams** are independent per purpose (`rng_for`). Production reused one seed for several draws: the
  single-span substitution length factor had correlation 1.0 with the edit position, and the l_m / l_i / l_t_m / l_t_i
  template was fully determined by position or mode. Distributions are unchanged but individual draws differ.
- Location-reference subsets have no reverse instructions (the release is forward-only for them; production stage 1
  wrote unused reverses for some subsets).
- Template placeholders are unified (`{x}`, `{y}`, `{ord}`, `{t}`, `{ts}`, `{te}`); rendered strings are identical.

### 9.5 annotate: single-span and multi-span

- zh content tokens keep Latin runs whole (1 multi-span placement difference in 3,000 zh rows).
- Single-span insertion reverses use the released partial-anchor form directly (production stage 1 was always
  anchored and the release rewrote it); a draw whose anchored form is also ambiguous is redrawn (production never
  rejected single insertions for this).
- Multi-span instructions are produced in the released form: forward = text-order join with `INSTRUCTION_SEP` (the HF
  `forward_instruction`), reverse = shuffled, partially de-anchored sub-reverses joined the same way (the trained form);
  rows whose anchored reverse is ambiguous are dropped, as in the release. Production stage 1 stored chunk lists with
  the reverses right to left. As in production, a strictly sequential reading of the shuffled reverse (each
  sub-reverse applied to the text left by the previous one) can fail: a deletion context can reach into another edit's
  new text, or reappear once an earlier-listed reverse restores its target (context ‘著名’ after ‘于著名’ is restored).
  This affected about 4% of the multi-span records of a 2,500-record run with stand-in LLM words and none of 37 with
  DeepSeek. Applied together to the edited text (§4.4), the reverse is unambiguous.
- Multi-span rows are also dropped when a substitution's new text does not occur exactly once in the edited text
  (§4.4), because its reverse names that text alone. The release rewrite checked only the reverses of insertions and
  kept such rows with an ambiguous reverse; with LLM-written words such collisions are rare.
- Insertion edits store the grown anchor in `target` (production stored an empty target and kept the anchor only in the
  instruction).
- The reverse-chain replay locates deletion contexts case-insensitively and whole-word, and re-inserts through
  `apply_edits`.
- The "no change" test compares content keys (production single deletion compared raw strings).
- Kept on purpose: single deletion has no adjacent-duplicate check; the multi-span duplicate check is sentence-level;
  no re-capitalisation in en after head edits ("Of course You can").
- With production random draws injected (including its `int(random() * n)` index draw, see §9.4), zh selections are
  identical to production (deletion 2,805 / 2,805, insertion 2,690 / 2,690, multi-span 2,704 / 2,706); every en
  difference is explained by the uniqueness semantics above.

### 9.6 annotate: location-reference

- The alignment gate applies to all twelve subsets (production gated only d4, so d1/m1/m2/i1/i2/l_\*/l_t_\* sent rows
  with misaligned text to synthesis). On the first 3,000 production d1 records per language the gate drops 139 en rows
  (138 by the shared-token check, 1 too short, no token mismatch) and 695 zh rows (694 with English words).
- d1/m1/i1 instructions show the surface text of the edited occurrences (production showed a normalized form: en
  lowercased, zh Latin without spaces, e.g. `prec轮融资`); the same target is sent to the LLM.
- The ordinal visibility check (d4/m2/i2) counts the surface text with `count_occurrences`; zh repeats with embedded
  Latin letters become eligible (5 / 2,052 zh d4 rows choose a different repeat).
- d1/m1/i1 require the same visible count (production gated d4 only). Without the check, an "all occurrences"
  instruction left the target in the text whenever jieba merged an occurrence into a longer word (它会) or an en word
  hid in a contraction (we're): on 2,000 gated rows about 2.4% of the d1/m1/i1 candidates (2.9% zh, 2.0% en) are now
  dropped or move to another repeat.
- The cap of 8 occurrences applies to m2 and i2 too (production capped d4 only; m2/i2 ordinals reached 25 / 85).
- m1/m2/i1/i2 reject new adjacent duplicates, and m2 also checks reversibility (production had neither).
- d1 skips a repeat when deleting all occurrences leaves no content (production checked for an empty string, so
  punctuation-only leftovers survived).
- One bucket table (9 buckets) for all occurrence subsets (production d4 used 7) and shared stratified sampling
  without reservoir quotas.
- **Time disambiguation** (new): l_t_\* candidates are dropped unless the shown time(s) map back to exactly the edited
  words (3.8% zh / 10.2% en of gated rows).
- l_t_m / l_t_i reject new adjacent duplicates and check reversibility (production checked only the validator and
  "no change"); l_m uses the global reversibility check (production's positional check could never fail).
- The l_d "no change" check is gone (it cannot fire when at least two tokens are kept).
- `meta` is reduced to the fields listed in §4; `affected_timestamps` and `span_uncovered` are dropped because the gate
  guarantees token alignment.
- Kept from production: no function-word stop list (repeats such as 的 / the are common); the
  `GLOBAL_SUBSTITUTION_PROMPT` says "every occurrence" also for single-position edits; en insertions before a
  sentence-initial word keep its capital; d1/d4 deletions may create adjacent duplicates; zero-length shown intervals
  ("between 3.8s and 3.8s", 0.7% zh / 2.0% en of span records) are kept.

### 9.7 synthesize

- The language comes from each record (production used a per-file flag), and the diff uses the gate tokenizer, which
  follows the aligner (§4.1) where production's synthesizer split a word further. On 64,000 production records the
  plans are identical in 62,618 cases. 1,002 differ because the original text has a word that production split and the
  aligner keeps whole (a hyphen, `5:00`, `U.S.`), so production regenerated speech that did not change; 609 of these
  rows pass our gate. 379 differ because the new text contains such a word, which now counts as one token and gets a
  shorter target duration ("Spider-Man" for "Daredevil": 0.64 s instead of 1.28 s).
- The edit plan comes from a global diff of `align_word` against `edited_text` (as in production), not from the
  annotated `edits`. When the edited unit has an identical neighbouring copy, the diff may regenerate a different,
  content-equivalent occurrence than the instruction names: the edited speech is still correct, but `edit_regions`
  can point at the other copy. On the annotate outputs of 4,416 d4 and 2,371 d1 records, 216 d4 records (141
  regenerating an earlier copy, 75 a later one) and 4 d1 records have an annotated deletion / substitution span less
  than half covered by the plan; the released data behaves the same way.
- No spoken change gives status `no_change` and no audio (production regenerated the whole utterance and still wrote
  `ok`). The empty-`align_word` fallback is gone (the gate guarantees alignment).
- `too_long` is an explicit status (production swallowed the DiT error and lost the record, 3,474 items); other decode
  errors raise instead of being swallowed.
- `no_context` is an explicit status: when the widened mask covers the whole condition, production sampled with no
  acoustic prompt and wrote `ok` (a different voice, e.g. sim -0.02, dropped only by the SIM threshold). On the
  production multi-span annotate outputs this is 1.95% of multi_EN and 1.96% of multi_ZH rows (0.48% of all
  single-span and multi-span rows). These records are not synthesized, so they never reach ASR or SIM, and they cannot
  pass whatever `--min-sim` is set to.
- Deterministic per-record seeds (production: unseeded), so results do not depend on sharding.
- Wavs are always PCM_16 (production wrote FLOAT32 or PCM_16 depending on the machine) and are scaled down to peak 1
  when Vocos overshoots full scale, as the release encoder did for FLOAT wavs (production FLOAT wavs kept the
  overshoot; about 25% peaked above 1.0). They are stored as `<audio-dir>/<task>/<subset>/<uuid>.wav` (production: one
  flat directory with about 300k files); one output jsonl per run, no metafile / summary / rolling merge.
- The output keeps the whole edit record; production's `source_parts_to_edit` / `parts_to_edit` become `edit_regions`
  (rounded as in the HF export); `fix_duration`, `warnings` and `spoken_delta` are not written.
- mp3 files are decoded by torchaudio's default backend (ffmpeg here). Production machines used different backends;
  ffmpeg and mpg123 decode the 32 kHz zh mp3s identically but differ slightly on 24 kHz en mp3s.
- Unchanged: 0.5 s mask widening, 0.1 s deletion filler, 0.18 s floor, speech-rate clamps, merge / shift rules, frame
  rounding, RMS handling, full edited-text pinyin condition, NFE 32 / CFG 2 / sway -1, fp16 EMA weights, full-utterance
  Vocos decoding without paste-back.

### 9.8 verify and export

- ASR runs the qwen-asr transformers backend in-process (batch 32) instead of vLLM servers; audio is decoded with
  librosa and peak-normalized above 1.0. On 420 location-reference pairs that production kept (70 each of d1 zh / en,
  l_m zh, l_t_i en, m1 en and i2 zh), 335 transcripts (80%) are identical to production's. The others differ in
  punctuation and also in words: numerals for number words ('eighty' / '80'), dropped fillers ('Uh') and repeats,
  near-homophones ('地' / '的') and substitutions ('怎么' / '什么'). 23 of the 420 (5.5%) would now be dropped as
  `asr_mismatch`, 10 / 70 (14%) on m1 en; with `--batch-size 1` it is 24, so batching is not the cause. The sample has
  only pairs that production kept, so it does not show pairs production dropped that would now pass. The scoring
  itself is bit-exact (re-scoring production transcripts reproduces 420 / 420 scores). The per-token diff list is not
  stored, and the field is `asr_text`.
- Padded batched bf16 decoding makes `asr_text` depend slightly on the batch composition, hence on `--batch-size` and
  `--num-shards`: between batch 32 and batch 1, 4 of the first 96 mini transcripts differ (punctuation only) and 32 of
  the 420 production pairs above (3 keep / drop decisions flip). Use `--batch-size 1` for bit-reproducible transcripts.
  `align` decodes in batches the same way.
- SIM is computed per pair on whole utterances (batch size 1, the F5-TTS evaluation recipe); production used
  length-bucketed, padded, masked batches. On the same 420 pairs: mean |difference| 0.004, max 0.095 (the largest on
  d1 rows whose synthetic wav is less than half as long as the original), all on the same side of 0.5. The field is
  `sim`, and the checkpoint must contain every model weight.
- Rows whose status is not `ok` skip ASR and SIM (production scored every row); an `ok` row whose wav cannot be
  transcribed (after the per-clip retry) or loaded is left undone, every other row is written, and the run fails at
  the end; any other error raises immediately.
- Export filters stored scores and writes a pairs file instead of deleting wavs. It keeps `sim > 0.5` (the production
  wav-deletion scripts kept `sim >= 0.5`, and some zh multi-span wavs with `asr_score <= 0.2` stayed on disk until the
  release export required 0) and `synthetic_duration <= 30` (the multi-span release export used `< 30`).

### 9.9 Mini demo

The 20 clips in `examples/mini/` were hand-picked with stricter filters than production applied (duration 4-10 s,
DNSMOS >= 3.5, zero WER between the Emilia and ASR transcripts, clean character set, one clip per shard and speaker; see
`examples/mini/README.md`).
`scripts/run_mini_pipeline.sh` replaces the production orchestration (8-GPU sharding, cluster submission, separate
environments) with one single-GPU flow.
