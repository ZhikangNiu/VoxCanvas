def length_hint(target_length: int | None, lang: str) -> str:
    """Soft length guidance injected as {length_hint}; flexible wording when no target is given."""
    if not target_length:
        return "- The length is flexible; longer or shorter is both fine, as long as it reads naturally."
    unit = "word(s)" if lang == "en" else "Chinese character(s)"
    return (
        f"- Length guidance: aim for roughly {target_length} {unit}; a little more or fewer is "
        f"fine — keep it natural and fluent rather than forcing the exact count."
    )


# System message sent with every prompt; without it deepseek-chat answers English insertion prompts in Chinese.
LANGUAGE_HINT = {
    "en": "The input sentence is English, so answer in English.",
    "zh": "The input sentence is Chinese, so answer in Chinese.",
}


# single_span / multi_span substitution; placeholders {raw_text} {target} {length_hint}.
SUBSTITUTION_PROMPT = """
# Task
You are given `raw_text` and a `target` word/phrase that appears in it. Output exactly ONE new word/phrase that will replace the `target`.
The result is consumed by a downstream TTS speech-editing system, so the replacement must be pronounceable and must produce a clearly audible pronunciation change when spoken.

# Output language
- The replacement MUST be in the same language as `target` (Chinese target -> Chinese replacement; English target -> English replacement). Do not switch language unless `target` itself is a foreign-language insert.

# Replacement rules
- Must be non-empty and DIFFERENT from `target`.
- Must be pure pronounceable content: at least one Chinese character / English letter / digit, and NO leading or trailing punctuation, symbols, or whitespace.
  Good: 上海 / 六个月 / banana / tomorrow morning
  Bad:  ，上海 / 上海。 / "banana" / ，
- Must cause an audible pronunciation change. Prefer changing the entity / number / object / action / attribute, so that the resulting sentence reads naturally and fluently.
{length_hint}
- Do NOT produce homophone / near-homophone / spelling-only / case-only / written-form variants that sound the same or nearly the same:
  - Chinese: avoid “盈利/营利”, “账户/帐户”, “其它/其他”, “做/作”, “的/地/得”.
  - English: avoid “color/colour”, “organization/organisation”, “OpenAI/open ai”.
  Good: 北京->上海 / 三个月->十二个月 / 苹果->香蕉 / 上涨->下跌
- IMPORTANT: the replacement MUST NOT be a word/phrase that ALREADY appears anywhere in `raw_text`. Pick something not present in `raw_text`, so that swapping it back later is unambiguous.
- Mentally verify that replacing `target` with your word in `raw_text` still yields fluent, natural text. If not, choose a different replacement.

# Input
raw_text: {raw_text}
target: {target}

# Output
Output ONLY the replacement word/phrase itself as a single line of plain text.
- Do not output JSON, Markdown, code fences, quotes around the word, labels, the original `target`, the full sentence, or any explanation.
- Do not prefix with "replacement:", "输出:", "Answer:", or similar.
- Do NOT include any reasoning, thinking, planning, analysis, or self-talk before or after the word.
- If you internally reason, keep it strictly internal; only emit the final replacement word.

Bad examples (do NOT do this):
让我想想，把‘北京’换成‘上海’比较合适，所以输出：上海
<think>target=北京</think>上海
"上海"
把‘北京’改成‘上海’

Good examples:
上海
banana
"""


# single_span / multi_span insertion; placeholders {raw_text} {target} {length_hint} {direction} {direction_upper} {direction_zh} {splice_order}.
INSERTION_PROMPT = """
# Task
You are given `raw_text` and a `target` word/phrase that appears in it.
Output exactly ONE short new word/phrase that will be inserted immediately {direction_upper} the `target` (加在 target 的{direction_zh}).
The result is consumed by a downstream TTS speech-editing system, so the inserted text must be pronounceable and must produce a clearly audible change when spoken.

# Direction
- The reader will splice your phrase {direction_upper} `target`, so it must read fluently in that exact position ({splice_order}).

# Output language
- The inserted text MUST be in the same language as `target` (Chinese target -> Chinese; English target -> English). Do not switch language unless `target` itself is a foreign-language insert.

# Insertion rules
- Must be non-empty, pure pronounceable content: at least one Chinese character / English letter / digit, and NO leading or trailing punctuation, symbols, or whitespace.
  Good: 非常 / 其实 / 每天 / quickly / really / last year
  Bad:  ，非常 / 非常。 / "really" / ，
{length_hint}
- Keep it a meaningful fragment: do NOT insert a whole clause or a long sentence.
- It must form natural, fluent text when placed {direction_upper} `target`.
- Do NOT insert a dangling / unfinished fragment. Do not end with words that strongly require a following complement.
  Bad Chinese endings: 由/被/把/向/对/从/给/为了/因为/所以/但是/包括/的.
  Bad English endings: by/to/from/with/because/although/including/the/a/an.
- IMPORTANT (reversibility): the inserted text MUST NOT be a word that ALREADY sits immediately {direction_upper} `target` anywhere in `raw_text`. Pick something that is not already {direction} `target`, so it can be unambiguously removed later.
- IMPORTANT: the inserted text MUST NOT contain `target` as a substring.
- Mentally verify: inserting your phrase {direction_upper} `target` yields fluent text, and later deleting your phrase from {direction} `target` restores `raw_text` exactly. If not, choose a different phrase.

# Input
raw_text: {raw_text}
target: {target}

# Output
Output ONLY the inserted word/phrase itself as a single line of plain text.
- Do not output JSON, Markdown, code fences, quotes around the phrase, labels, the original `target`, the full sentence, or any explanation.
- Do not prefix with "insert:", "输出:", "Answer:", or similar.
- After any internal reasoning, the FINAL line of your reply must be exactly the inserted phrase and nothing else.

Bad examples (do NOT do this):
让我想想，在‘北京’{direction_zh}加上‘非常’……所以输出：非常
"非常"
在‘北京’{direction_zh}加上‘非常’

Good examples:
非常
really
"""


# location_reference substitution (m1 m2 l_m l_t_m); placeholders {raw_text} {target}.
GLOBAL_SUBSTITUTION_PROMPT = """
# Task
You are given `raw_text` and a `target` word/phrase that appears in it. Output exactly ONE new word/phrase that will replace EVERY occurrence of `target`.
The result is consumed by a downstream TTS speech-editing system, so the replacement must be pronounceable and must produce a clearly audible pronunciation change when spoken.

# Output language
- The replacement MUST be in the same language as `target` (Chinese target -> Chinese replacement; English target -> English replacement). Do not switch language unless `target` itself is a foreign-language insert.

# Replacement rules
- Must be non-empty and DIFFERENT from `target`.
- Must be pure pronounceable content: at least one Chinese character / English letter / digit, and NO leading or trailing punctuation, symbols, or whitespace.
  Good: 上海 / 六个月 / banana / tomorrow morning
  Bad:  ，上海 / 上海。 / "banana" / ，
- Must cause an audible pronunciation change. Prefer changing the entity / number / object / action / attribute. Keep length roughly comparable to `target`; do not pile up extra modifiers to inflate length.
- Do NOT produce homophone / near-homophone / spelling-only / case-only / written-form variants that sound the same or nearly the same:
  - Chinese: avoid “盈利/营利”, “账户/帐户”, “其它/其他”, “做/作”, “的/地/得”.
  - English: avoid “color/colour”, “organization/organisation”, “OpenAI/open ai”.
  Good: 北京->上海 / 三个月->十二个月 / 苹果->香蕉 / 上涨->下跌
- IMPORTANT: the replacement MUST NOT be a word/phrase that ALREADY appears anywhere in `raw_text`. Pick something not present in `raw_text`, so that swapping it back later is unambiguous.
- Mentally verify that replacing every `target` with your word throughout `raw_text` still yields fluent, natural text. If not, choose a different replacement.

# Input
raw_text: {raw_text}
target: {target}

# Output
Output ONLY the replacement word/phrase itself as a single line of plain text.
- Do not output JSON, Markdown, code fences, quotes around the word, labels, the original `target`, the full sentence, or any explanation.
- Do not prefix with "replacement:", "输出:", "Answer:", or similar.
- Do NOT include any reasoning, thinking, planning, analysis, or self-talk before or after the word.
- If you internally reason, keep it strictly internal; only emit the final replacement word.

Bad examples (do NOT do this):
让我想想，把‘北京’换成‘上海’比较合适，所以输出：上海
<think>target=北京</think>上海
"上海"
把‘北京’改成‘上海’

Good examples:
上海
banana
"""


# location_reference insertion (i1 i2 l_i l_t_i); placeholders {raw_text} {target} {direction} {direction_zh}.
LOCATION_INSERTION_PROMPT = """
# Task
You are given `raw_text`, a `target` word/phrase that appears in it, and an insertion `direction`.
Output exactly ONE short new word/phrase that will be inserted {direction} the `target` ({direction_zh}).
The result is consumed by a downstream TTS speech-editing system, so the inserted text must be pronounceable and must produce a clearly audible change when spoken.

# Direction
- direction = {direction}  (insert the new text immediately {direction} the `target`; Chinese: 加在 target 的{direction_zh}).
- The reader will splice your phrase {direction} `target`, so it must read fluently in that exact position.

# Output language
- The inserted text MUST be in the same language as `target` (Chinese target -> Chinese; English target -> English). Do not switch language unless `target` itself is a foreign-language insert.

# Insertion rules
- Must be non-empty, pure pronounceable content: at least one Chinese character / English letter / digit, and NO leading or trailing punctuation, symbols, or whitespace.
  Good: 非常 / 其实 / 每天 / quickly / really / last year
  Bad:  ，非常 / 非常。 / "really" / ，
- Keep it SHORT and meaningful: prefer 2-6 Chinese characters, or 1-4 English words. Do not insert a whole clause or a long sentence.
- It must form natural, fluent text when placed {direction} `target`. Inserting {direction} EVERY occurrence of `target` (or a specific one) must still read naturally.
- Do NOT insert a dangling / unfinished fragment. Do not end with words that strongly require a following complement.
  Bad Chinese endings: 由/被/把/向/对/从/给/为了/因为/所以/但是/包括/的.
  Bad English endings: by/to/from/with/because/although/including/the/a/an.
- IMPORTANT (reversibility): the inserted text MUST NOT be a word that ALREADY sits immediately {direction} `target` anywhere in `raw_text`. Pick something that is not already adjacent to `target` in that direction, so it can be unambiguously removed later.
- IMPORTANT: the inserted text MUST NOT contain `target` as a substring.
- Mentally verify: inserting your phrase {direction} `target` yields fluent text, and later deleting your phrase from {direction} `target` restores `raw_text` exactly. If not, choose a different phrase.

# Input
raw_text: {raw_text}
target: {target}
direction: {direction}

# Output
Output ONLY the inserted word/phrase itself as a single line of plain text.
- Do not output JSON, Markdown, code fences, quotes around the phrase, labels, the original `target`, the full sentence, or any explanation.
- Do not prefix with "insert:", "输出:", "Answer:", or similar.
- After any internal reasoning, the FINAL line of your reply must be exactly the inserted phrase and nothing else.

Bad examples (do NOT do this):
让我想想，在‘北京’后面加上‘非常’……所以输出：非常
"非常"
在‘北京’后面加上‘非常’

Good examples:
非常
really
"""
