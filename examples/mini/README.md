# Mini demo set

A small set of 20 Emilia clips (10 Mandarin, 10 English) used to run the whole VoxCanvas
content-editing pipeline end to end on a single GPU.

| file | content |
|---|---|
| `manifest.jsonl` | input of stage `align`: `{"uuid", "audio_path", "language"}` per clip |
| `audio/<uuid>.mp3` | the original Emilia mp3 files, copied unmodified |
| `aligned.jsonl` | precomputed output of stage `align` (Qwen3-ASR-1.7B text + Qwen3-ForcedAligner-0.6B `align_word`), used with `SKIP_ALIGN=1` |

## Selection

Clips were hand-picked from the Emilia `ZH` / `EN` subsets with these criteria:

- duration 4–10 s (actual 4.1–9.2 s, 135 s in total), DNSMOS ≥ 3.5;
- every clip comes from a different Emilia shard (`B000xx`) and speaker id;
- the Qwen3-ASR transcript agrees with the Emilia transcript (zero WER), contains only CJK
  characters (zh) or plain words (en) and standard punctuation, and passes the alignment gate
  (transcript tokens equal the forced-aligner tokens);
- at least 10 content tokens per transcript;
- 7 zh and 8 en clips contain a repeated, non-overlapping word or phrase whose visible count
  matches its token count (汽车, 的问题, 冒险 ×3, 妈妈 ×3, 库存 ×3, 贪婪, 太子; me, were found in,
  fuel, Larry, very ×3, people, himself, I feel proud), so the occurrence subsets (d1 d4 m1 m2 i1 i2)
  produce output;
- several clips start with a discourse word suited for sentence-boundary edits
  (所以, 但是, 如果, 其实, 因此; And, But, Now, Yeah, So).

| uuid | language | duration (s) | transcript |
|---|---|---|---|
| ZH_B00000_S05795_W000104 | zh | 7.15 | 从福特发明汽车以后，这一百年，汽车基本上没有改变。 |
| ZH_B00001_S05349_W000020 | zh | 5.40 | 所以我经常说啊，你不要把成长的问题归结为生意的问题。 |
| ZH_B00003_S01538_W000138 | zh | 5.71 | 追求梦想是要冒险的，但是人生最大的冒险就是不去冒险。 |
| ZH_B00005_S04929_W000003 | zh | 9.03 | 妈妈以前不是这样的。小凯的妈妈很漂亮，说话也很温柔。小凯非常喜欢妈妈。 |
| ZH_B00006_S04086_W000027 | zh | 4.60 | 但是，毫无疑问，这是英国逐步退出世界舞台中央的开始。 |
| ZH_B00007_S02458_W000038 | zh | 4.09 | 如果一种物质里面有燃素，那么它就可以被点着。 |
| ZH_B00008_S04728_W000014 | zh | 7.10 | 从库存周期的角度看，工业企业仍处于主动去库存阶段，库存去化尚未见底。 |
| ZH_B00009_S05190_W000036 | zh | 5.48 | 其实，这种整合你已经在电话语音服务这一简单的平台上体验过了。 |
| ZH_B00010_S06560_W000077 | zh | 7.91 | 因此啊，在贪婪之前，我们必须经过思考，确保我们的贪婪不会让我们堕入深渊。 |
| ZH_B00011_S05354_W000006 | zh | 7.23 | 皇上对太子已经产生了猜忌之心，太子的日子好过不了了。 |
| EN_B00000_S00247_W000001 | en | 7.75 | And apparently, I'm supposed to draw something that makes me happy or reminds me of happiness. |
| EN_B00001_S07129_W000105 | en | 7.47 | As per this report, yes, the highest cases were found in Delhi, but the lowest cases amongst all the states were found in Gujarat. |
| EN_B00002_S09489_W000014 | en | 6.83 | Fuels made from oil, such as gasoline, diesel fuel, and jet fuel, power our cars, trucks, and airplanes. |
| EN_B00003_S00321_W000006 | en | 6.21 | Larry often outplayed Michael, but that changed when Michael suddenly grew much taller than Larry. |
| EN_B00004_S05804_W000003 | en | 6.25 | Books about the law are notorious for being very long, very dull, and very difficult to read. |
| EN_B00005_S04997_W000004 | en | 8.29 | And of course, the force in Star Wars makes its successful users strong beyond belief. |
| EN_B00006_S04542_W000109 | en | 9.18 | Now this is a discussion about how to optimize your brain. Many people who wake up quickly and just naturally feel like bouncing out of bed. I envy these people. |
| EN_B00008_S00120_W000706 | en | 6.13 | But Severus himself seemed undaunted, and readied himself for what he was confident would be his final test. |
| EN_B00010_S03130_W000005 | en | 8.64 | Yeah, I feel proud. I feel proud that I stood by my convictions. I know that a lot of people don't want to hear. |
| EN_B00011_S09741_W000000 | en | 4.31 | So this was an opportunity to really change the direction. |

Transcripts are the Qwen3-ASR-1.7B outputs of stage `align` (the text the edits are made on); the
original Emilia transcripts may differ in punctuation.

## Running the pipeline

From the repository root, with the project environment active (or prefix the command with `uv run`) and the
checkpoints in `checkpoints/` (see [Checkpoints](../../README.md#checkpoints)):

```bash
bash scripts/run_mini_pipeline.sh
```

Stages: `align` → `annotate` (all 16 subsets) → concatenation into `edits.jsonl` → `synthesize` →
`verify.asr` → `verify.sim` → `export`. The align, annotate, synthesize and verify stages skip rows
already present in their outputs, so the script can be re-run after an interruption.

Environment variables:

| variable | default | meaning |
|---|---|---|
| `OUTPUT_DIR` | `outputs/mini` | output directory |
| `MANIFEST_JSONL` | `examples/mini/manifest.jsonl` | manifest passed to `align` (set it to run on your own data) |
| `SKIP_ALIGN` | `0` | `1` = skip `align` and start from `$ALIGNED_JSONL` |
| `ALIGNED_JSONL` | `examples/mini/aligned.jsonl` | aligned rows used with `SKIP_ALIGN=1` |
| `F5_CKPT` | unset | F5-TTS checkpoint, passed as `synthesize --ckpt` |
| `VOCODER` | unset | Vocos vocoder (local directory or `charactr/vocos-mel-24khz`), passed as `synthesize --vocoder` |
| `ASR_MODEL` | unset | ASR model, passed as `align --asr-model` and `verify.asr --model` |
| `ALIGNER_MODEL` | unset | forced aligner, passed as `align --aligner-model` |
| `SIM_CKPT` | unset | WavLM-Large speaker-verification checkpoint (see [Checkpoints](../../README.md#checkpoints)), passed as `verify.sim --ckpt` |
| `LLM_BASE_URL` | unset | OpenAI-compatible endpoint, passed as `annotate --base-url` |
| `LLM_MODEL` | unset | LLM model name, passed as `annotate --model` |
| `LLM_API_KEY_ENV` | unset | name of the environment variable (or `.env` entry) holding the API key, passed as `annotate --api-key-env` |
| `LLM_EXTRA_BODY` | unset | JSON passed as `annotate --extra-body`, e.g. `{"top_k": 20, "chat_template_kwargs": {"enable_thinking": false}}` for a self-hosted Qwen3 server (`top_k` 20 as in the release; see the top-level README, "LLM endpoint") |
| `PYTHON` | `python` | Python interpreter |

An unset optional variable is not passed, so the stage uses its own default (the checkpoints in `checkpoints/` and the
DeepSeek endpoint listed in the top-level README; see `python -m voxcanvas.<stage> --help`). With the default endpoint,
the annotate stage needs the API key, e.g. `DEEPSEEK_API_KEY=...` in `.env` (see `.env.example`).
The GPU stages run on the first visible GPU; select it with `CUDA_VISIBLE_DEVICES`.

Outputs in `$OUTPUT_DIR`: `aligned.jsonl` (unless `SKIP_ALIGN=1`), `edits/<task>_<subset>.jsonl`, `edits.jsonl`,
`audio/<task>/<subset>/<uuid>.wav`, `synth.jsonl`, `asr.jsonl`, `sim.jsonl` and the final pairs `pairs.jsonl`.

## Attribution and license

The audio clips and their ids come from the Emilia dataset:

> Haorui He, Zengqiang Shang, Chaoren Wang, Xuyuan Li, Yicheng Gu, Hua Hua, Liwei Liu, Chen Yang, Jiaqi Li,
> Peiyang Shi, Yuancheng Wang, Kai Chen, Pengyuan Zhang, Zhizheng Wu.
> "Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation."
> IEEE Spoken Language Technology Workshop (SLT), 2024.
> https://arxiv.org/abs/2407.05361, dataset: https://huggingface.co/datasets/amphion/Emilia-Dataset

Emilia is released under the Creative Commons Attribution-NonCommercial 4.0 International license
(CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/). The clips in `audio/` are
redistributed unmodified under that license, and the Emilia ids, transcripts and word timestamps in
`manifest.jsonl` and `aligned.jsonl` are derived from it, so all of these files may be used for
non-commercial purposes only. The MIT license of the code in this repository does not apply to them;
see [LICENSE](LICENSE) in this directory for the data license, the Emilia copyright notice and the changes.
