# Mini nonverbal demo set

A small set of 20 NonVerbalSpeech-38K clips (10 Mandarin, 10 English), copied unmodified, used by
`scripts/run_mini_nonverbal.sh` and the tests to run the VoxCanvas nonverbal-editing pipeline end to end on a
single GPU.

| file | content |
|---|---|
| `manifest.jsonl` | input of stage `align`: `{"uuid", "audio_path", "language", "transcript"}` per clip |
| `audio/<uuid>.wav` | the original NonVerbalSpeech-38K wav files (24 kHz, mono, 16-bit PCM), copied unmodified |
| `aligned.jsonl` | precomputed output of stage `align` (tag-stripped transcript + word timestamps), used with `SKIP_ALIGN=1` |

## Selection

Clips were hand-picked from the NonVerbalSpeech-38K `ZH` / `EN` subsets with these criteria:

- duration 4–10 s (actual 4.590–9.517 s, 138.9 s in total);
- exactly one nonverbal tag, placed mid-sentence (speech before and after it), and no `<B>` / `</B>` overlap markup;
- the tag span found by forced alignment overlaps the dataset's detected `non_verbal_region`, and the region's
  centre lies in the middle 70 % of the clip;
- each language covers all 10 NonVerbalSpeech-38K tag categories once (sigh, breath, laugh, cough, crying, sniff,
  gasp, snore, throatclearing, yawn);
- the caption is unique in the dataset, and zh captions contain no Latin letters;
- no offensive content.

| uuid | language | duration | transcript |
|---|---|---|---|
| nvs38k_0012480 | zh | 8.201 | 啊，奥古斯丁跟我说了，晚安，他还要彻夜照顾表哥，[sigh]我也好，想生个病。 |
| nvs38k_0017913 | zh | 4.590 | 大概连我也快要忘了[breath]当年冥镜台发生的一切了。 |
| nvs38k_0018378 | zh | 6.244 | 走的这么急嘛，人狠话不多，还不爱洗澡，[laugh]太喜欢这样的性格了。 |
| nvs38k_0026954 | zh | 6.615 | 啊，根据角色，然后尽量去贴近角色，尽量给自己营造这种氛围嗯，[cough]理解理解角色。 |
| nvs38k_0027371 | zh | 6.429 | 我被喂下了血骨，还莫名顶替了别人的剧情。[crying]我怎么那么惨呢？ |
| nvs38k_0029409 | zh | 9.314 | 那是[sniff]传说中的此莲真火天地真火榜排名第一百零六。 |
| nvs38k_0029862 | zh | 7.122 | 全盲已经被我调到了南阳战区，今天[gasp]你不可能再见到他了。 |
| nvs38k_0030188 | zh | 6.885 | 届时来世只会更加的凶猛，[snore]你们是必死无疑呀。 |
| nvs38k_0030986 | zh | 5.535 | 可以活动身体，[throatclearing]不能每天老练练太极拳，我也可以打打网球。 |
| nvs38k_0037647 | zh | 4.961 | 这场行动我是不会参加的，[yawn]这件事是最后一次。 |
| nvs38k_0001377 | en | 7.071 | I'm sure that Uncle Sam will have this all sorted out by tomorrow and welcomes the United States. Almost. [laugh] Thank you. |
| nvs38k_0008760 | en | 7.543 | If I wasn't here to tell you, my parents are going to Portugal. [sigh] Oh! We love Portugal! |
| nvs38k_0016614 | en | 6.497 | when you find the Colonel infiltrate his team by [throatclearing] whatever means available. And, |
| nvs38k_0023539 | en | 8.572 | Lady listen to miss [cough] miss. I need you to open your eyes for me. All right Come on now. Oh, yeah, you can do that from the kitchen. Come on |
| nvs38k_0025998 | en | 6.868 | Friends of the Library is holding an historic manuscript viewing. [sniff] Friends of the Library. |
| nvs38k_0029171 | en | 5.012 | Oh, right, the bungalow. [snore] You don't look so fierce. |
| nvs38k_0031504 | en | 7.998 | Ha! Instead of TV, I typed TB. [crying] Let me Google that. Turn off the devices. |
| nvs38k_0033405 | en | 8.775 | I'm dead to everyone unless I try to become what I may be. [breath] Only the water of life will free what can save us. |
| nvs38k_0037093 | en | 5.133 | Mina, [gasp] stay with me, please. |
| nvs38k_0037916 | en | 9.517 | singing a slow, sad song about a dying wizard called Odo. Ah, [yawn] the good die young, muttered Hagrid, |

The uuid `nvs38k_<n>` is row `n` (7 digits) of the dataset's train split. Transcripts are the NonVerbalSpeech-38K captions with the nonverbal tag in place; the tags `[laughing]` and
`[coughing]` are renamed to `[laugh]` and `[cough]`, everything else is unchanged.

## Running the pipeline

From the repository root, with the project environment active (or prefix the command with `uv run`) and the
checkpoints in `checkpoints/` (see [Checkpoints](../../README.md#checkpoints)):

```bash
bash scripts/run_mini_nonverbal.sh
```

The captions are not Qwen3-ASR transcripts, so the exact-match ASR filter of `export` drops pairs whose speech is
fine (e.g. 啊 / 哈, Mina / Minna); `MAX_ASR_SCORE=0.1 bash scripts/run_mini_nonverbal.sh` relaxes it (see
[docs/pipeline.md](../../docs/pipeline.md) §6).

## Attribution and license

The audio clips, their ids and captions come from the NonVerbalSpeech-38K (NVS-38K) dataset:

> NonVerbalSpeech-38K: A Scalable Pipeline for Enabling Non-Verbal Speech Generation and Understanding.
> Dataset: https://huggingface.co/datasets/nonverbalspeech/nonverbalspeech38k,
> demo page: https://nonverbalspeech38k.github.io/nonverspeech38k/

NonVerbalSpeech-38K is released under the Creative Commons Attribution-NonCommercial 4.0 International license
(CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/). The clips in `audio/` are redistributed
unmodified under that license, and the ids and transcripts in `manifest.jsonl` and the tag-stripped transcripts
and word timestamps in `aligned.jsonl` are derived from it, so all of these files may be used for non-commercial
purposes only. The MIT license of the code in this repository does not apply to them; see [LICENSE](LICENSE) in
this directory for the data license, the dataset disclaimer and the changes.

Disclaimer reproduced from the NonVerbalSpeech-38K dataset card:

> Disclaimer: The NVS-38K dataset does not own the copyrights of the audio files it contains; copyright ownership
> resides with the original creators. The dataset is made available solely for non-commercial research purposes
> under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.
