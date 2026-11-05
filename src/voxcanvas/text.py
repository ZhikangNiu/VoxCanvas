import logging
import re
import unicodedata
from itertools import pairwise

import jieba


jieba.setLogLevel(logging.WARNING)

CJK = "\u4e00-\u9fff\u3400-\u4dbf"
KANA_HANGUL = "\u3041-\u3096\u30a1-\u30fa\uac00-\ud7a3"  # kana letters and hangul syllables, no kana punctuation
# Scripts the language label rules out; a row written in one is a mislabelled utterance of another language.
FOREIGN_SCRIPT_RE = {"en": re.compile(f"[{CJK}{KANA_HANGUL}]"), "zh": re.compile(f"[{KANA_HANGUL}]")}
CONTENT_CHAR_RE = re.compile(f"[0-9A-Za-z{CJK}]")
CJK_RE = re.compile(f"[{CJK}]")
NON_CONTENT_RE = re.compile(f"[^0-9a-z{CJK}]")
# English word (inner ' ’ - kept) or a single CJK character.
WORD_RE = re.compile(f"[A-Za-z0-9]+(?:['’\\-][A-Za-z0-9]+)*|[{CJK}]")
# Latin/digit run embedded in Chinese text; tokenized with WORD_RE instead of jieba (keeps "It's" whole).
LATIN_RUN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9'’\- ]*[A-Za-z0-9])?")
# Forced-aligner tokens: each CJK character and each run of other non-space characters (zh also splits Latin letters).
ALIGNED_PIECE_RE = {
    "en": re.compile(f"[{CJK}]|[^\\s{CJK}]+"),
    "zh": re.compile(f"[{CJK}A-Za-z]|[^\\s{CJK}A-Za-z]+"),
}
DUP_PUNCT_RE = re.compile(r"[，,、。.！!？?；;：:]\s*(?=[，,、。.！!？?；;：:])")
LEADING_PUNCT_RE = re.compile(r"^[\s，,、。.！!？?；;：:）)】」』\]]+")
ZH_DUPLICATE_RE = re.compile(f"([{CJK}]{{1,4}})\\1")
EN_DUPLICATE_RE = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
# Characters that sound (nearly) the same in TTS; swapping only these gives no audible edit.
HOMOPHONE_GROUPS = ("他她它", "的地得", "在再", "做作", "座坐", "那哪", "了啦", "吗嘛", "呢呐", "和合")
EDIT_OPS = ("deletion", "insertion", "substitution")
MIN_CONTENT_TOKENS = 4  # content tokens a row needs to pass the gate


def tokenize_content(text: str, lang: str) -> list[tuple[str, int, int]]:
    """Edit units with char offsets: jieba words for zh (embedded Latin/digit runs split by WORD_RE), WORD_RE for en.

    Tokens without a letter, digit or CJK character (punctuation, spaces) are dropped.
    """
    if lang == "en":
        pieces = [(0, len(text), True)]
    else:
        pieces, pos = [], 0
        for run in LATIN_RUN_RE.finditer(text):
            pieces += [(pos, run.start(), False), (run.start(), run.end(), True)]
            pos = run.end()
        pieces.append((pos, len(text), False))
    tokens = []
    for start, end, is_latin in pieces:
        piece = text[start:end]
        found = (
            ((m.group(), m.start(), m.end()) for m in WORD_RE.finditer(piece)) if is_latin else jieba.tokenize(piece)
        )
        tokens += [(word, start + s, start + e) for word, s, e in found if CONTENT_CHAR_RE.search(word)]
    return tokens


def segment_ids(text: str, tokens: list[tuple[str, int, int]]) -> list[int]:
    """Segment index per token; a new segment starts whenever the gap before a token contains punctuation."""
    ids, segment = [], 0
    for i, (_, start, _) in enumerate(tokens):
        if i and text[tokens[i - 1][2] : start].strip():
            segment += 1
        ids.append(segment)
    return ids


def tokenize_aligned(text: str, lang: str) -> list[tuple[str, int, int]]:
    """Tokens as the forced aligner emits them, with offsets from the first to the last kept character.

    A token keeps only the letters, digits and ' of its piece (ALIGNED_PIECE_RE), lowercased (`5:00` -> "500",
    `peer-reviewed` -> "peerreviewed", `it’s` -> "its"); zh also splits Latin letters (docs/pipeline.md §4.1).
    """
    tokens = []
    for m in ALIGNED_PIECE_RE[lang].finditer(text):
        kept = [i for i in range(m.start(), m.end()) if text[i] == "'" or unicodedata.category(text[i])[0] in "LN"]
        if kept:
            tokens.append(("".join(text[i] for i in kept).lower(), kept[0], kept[-1] + 1))
    return tokens


def normalize_align_tokens(align_word: list[list]) -> list[str]:
    return [word[0].strip().lower() for word in align_word]


def gate(row: dict) -> bool:
    """True if the text has no "[UNK]", is in the script of its language, matches `align_word` and is long enough.

    No aligned token may be shared by two content tokens (`U.S.`, `5:00`), so each content token has its own time span.
    """
    text, lang = row["text"], row["language"]
    if lang not in ("zh", "en"):
        raise ValueError(f"Unsupported language {lang!r} for {row['uuid']}")
    if "[UNK]" in text or FOREIGN_SCRIPT_RE[lang].search(text):
        return False
    aligned = tokenize_aligned(text, lang)
    if [token for token, _, _ in aligned] != normalize_align_tokens(row["align_word"]):
        return False
    content = tokenize_content(text, lang)
    is_shared = any(
        sum(token_start < end and token_end > start for _, token_start, token_end in content) > 1
        for _, start, end in aligned
    )
    return len(content) >= MIN_CONTENT_TOKENS and not is_shared


def content_key(text: str) -> str:
    """Lowercased text restricted to [0-9a-z] and CJK; equal keys mean equal spoken content."""
    return NON_CONTENT_RE.sub("", text.lower())


def apply_edits(text: str, edits: list[dict], lang: str) -> str:
    """Apply non-overlapping edits ("op", "new", "direction", "region" on `text`), then tidy spaces and punctuation.

    The tidying rules are listed in docs/pipeline.md §4.1 (edit application).
    """
    ordered = sorted(edits, key=lambda edit: edit["region"][0])
    for left, right in pairwise(ordered):
        assert left["region"][1] <= right["region"][0], f"overlapping edits: {left} {right}"
    has_deletion = any(edit["op"] == "deletion" for edit in edits)
    fix_spaces = lang == "en" and any(edit["op"] != "substitution" for edit in edits)
    at_head = ordered[0]["op"] == "deletion" and not CONTENT_CHAR_RE.search(text[: ordered[0]["region"][0]])
    for edit in reversed(ordered):
        assert edit["op"] in EDIT_OPS, edit
        start, end = edit["region"]
        if edit["op"] == "deletion":
            if lang == "en" and end < len(text) and text[end] == " ":
                end += 1
            elif lang == "en" and start > 0 and text[start - 1] == " ":
                start -= 1
            text = text[:start] + text[end:]
        elif edit["op"] == "substitution":
            text = text[:start] + edit["new"] + text[end:]
        elif edit["direction"] == "before":
            text = text[:start] + edit["new"] + (" " if lang == "en" else "") + text[start:]
        else:
            text = text[:end] + (" " if lang == "en" else "") + edit["new"] + text[end:]
    if fix_spaces:
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    if has_deletion:
        text = DUP_PUNCT_RE.sub("", text)
    if at_head:
        text = LEADING_PUNCT_RE.sub("", text.strip())
    return text.strip() if fix_spaces else text


def edit_points(text: str, edited_text: str, edits: list[dict]) -> list[int]:
    """Offset in `edited_text = apply_edits(text, edits, lang)` where each edit (sorted by region) took place.

    The offset is the end of the last content character before the edit point; apply_edits keeps content outside edits.
    """
    content_ends = [m.end() for m in CONTENT_CHAR_RE.finditer(edited_text)]
    points, delta = [], 0
    for edit in sorted(edits, key=lambda edit: edit["region"][0]):
        start, end = edit["region"]
        point = end if edit["op"] == "insertion" and edit["direction"] == "after" else start
        num_before = len(CONTENT_CHAR_RE.findall(text[:point])) + delta
        points.append(content_ends[num_before - 1] if num_before else 0)
        removed = 0 if edit["op"] == "insertion" else len(CONTENT_CHAR_RE.findall(text[start:end]))
        delta += len(CONTENT_CHAR_RE.findall(edit["new"])) - removed
    return points


def find_occurrences(text: str, phrase: str, lang: str) -> list[tuple[int, int]]:
    """Case-insensitive, possibly overlapping matches of `phrase`; in en a match may not extend a word on its sides."""
    assert phrase, "empty phrase"
    before = "(?<!\\w)" if lang == "en" and re.match(r"\w", phrase[0]) else ""
    after = "(?!\\w)" if lang == "en" and re.match(r"\w", phrase[-1]) else ""
    pattern = re.compile(f"{before}(?={re.escape(phrase)}{after})", re.IGNORECASE)
    return [(m.start(), m.start() + len(phrase)) for m in pattern.finditer(text)]


def count_occurrences(text: str, phrase: str, lang: str) -> int:
    return len(find_occurrences(text, phrase, lang))


def has_adjacent_duplicate(text: str, lang: str) -> bool:
    """zh: a 1-4 CJK-character string immediately repeated; en: the same word twice in a row (case-insensitive)."""
    return (EN_DUPLICATE_RE if lang == "en" else ZH_DUPLICATE_RE).search(text) is not None


def same_language(new: str, target: str, lang: str) -> bool:
    """en text must not contain CJK; zh text must contain CJK unless the target itself has none."""
    if lang == "en":
        return not CJK_RE.search(new)
    return bool(CJK_RE.search(new)) or not CJK_RE.search(target)


def has_edge_mark(text: str) -> bool:
    """True if `text` starts or ends with punctuation, a symbol, a space or a control character (Unicode P, S, Z, C)."""
    return unicodedata.category(text[0])[0] in "PSZC" or unicodedata.category(text[-1])[0] in "PSZC"


def is_valid_replacement(new: str, target: str, raw_text: str, lang: str) -> bool:
    """LLM replacement check: pronounceable, same language, no edge marks, not in raw_text, no homophone swap."""
    if not CONTENT_CHAR_RE.search(new) or has_edge_mark(new):
        return False
    if not same_language(new, target, lang) or new.lower() in raw_text.lower():
        return False
    if len(target) != len(new):
        return True
    diffs = [(a, b) for a, b in zip(target, new, strict=True) if a != b]
    is_homophone_swap = all(any(a in group and b in group for group in HOMOPHONE_GROUPS) for a, b in diffs)
    return not is_homophone_swap


def is_valid_insertion(new: str, anchor: str, raw_text: str, direction: str, lang: str) -> bool:
    """LLM insertion check: pronounceable, same language, no edge marks, does not contain the anchor.

    Also rejects `new` already next to the anchor on that side in raw_text (with or without a space, any case).
    """
    if not CONTENT_CHAR_RE.search(new) or has_edge_mark(new):
        return False
    if not same_language(new, anchor, lang) or anchor.lower() in new.lower():
        return False
    forms = (new + anchor, new + " " + anchor) if direction == "before" else (anchor + new, anchor + " " + new)
    return not any(count_occurrences(raw_text, form, lang) for form in forms)


def undo_insertion(text: str, anchor: str, new: str, direction: str, lang: str) -> str:
    """Simulate "delete the `new` before/after `anchor`" on the whole text (every adjacent pair, any case)."""
    gap = "\\s+" if lang == "en" else ""
    if direction == "before":
        pair = f"{re.escape(new)}{gap}({re.escape(anchor)})"
    else:
        pair = f"({re.escape(anchor)}){gap}{re.escape(new)}"
    if lang == "en":
        pair = f"(?<!\\w){pair}(?!\\w)"
    return re.sub(pair, lambda m: m.group(1), text, flags=re.IGNORECASE)


def edit_round_trips(raw: str, edited: str, target: str, new: str, op: str, direction: str, lang: str) -> bool:
    """True if `edited` changes raw, adds no new adjacent duplicate and the global reverse edit restores raw."""
    if edited == raw or (has_adjacent_duplicate(edited, lang) and not has_adjacent_duplicate(raw, lang)):
        return False
    if op == "substitution":
        restored = edited.replace(new, target)
    else:
        restored = undo_insertion(edited, target, new, direction, lang)
    return content_key(restored) == content_key(raw)


def edit_keeps_alignment(text: str, edited: str, edits: list[dict], lang: str) -> bool:
    """True if the aligned tokens of `edited` are those of `text` with `edits` applied token by token.

    False when applying the edits changes the aligner's tokenization outside their own regions, e.g. deleting 起 in
    zh "21起500" glues the two numbers into one token, so the edit covers speech the instruction does not name.
    """
    aligned = tokenize_aligned(text, lang)
    expected, k = [], 0
    for edit in sorted(edits, key=lambda edit: edit["region"][0]):
        start, end = edit["region"]
        while k < len(aligned) and aligned[k][2] <= start:
            expected.append(aligned[k][0])
            k += 1
        covered = []
        while k < len(aligned) and aligned[k][1] < end:
            covered.append(aligned[k][0])
            k += 1
        new = [token for token, _, _ in tokenize_aligned(edit["new"], lang)]
        if edit["op"] == "substitution":
            expected += new
        elif edit["op"] == "insertion":
            expected += new + covered if edit["direction"] == "before" else covered + new
    expected += [token for token, _, _ in aligned[k:]]
    return expected == [token for token, _, _ in tokenize_aligned(edited, lang)]
