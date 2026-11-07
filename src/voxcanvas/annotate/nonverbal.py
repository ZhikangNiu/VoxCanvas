import random
import re

from voxcanvas.annotate.llm import LLMClient
from voxcanvas.annotate.records import SENTENCE_LENGTH_BOUNDS, build_record, length_bucket
from voxcanvas.annotate.templates import ANCHORED_DELETION, INSERTION
from voxcanvas.text import CONTENT_CHAR_RE, NV_TAG_RE, strip_nv_tags, tokenize_aligned, tokenize_content


SUBSETS = ("insertion",)
NEEDS_LLM = frozenset()
ANCHOR_UNITS = {"en": 3, "zh": 4}  # anchor length before uniqueness growth: words (en) / characters (zh)
MIN_ZH_ANCHOR_CHARS = 2
MAX_EN_ANCHOR_WORDS = 6
# en anchor word: any Unicode letters / digits (café stays whole), inner ' ’ - kept.
EN_ANCHOR_WORD_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")


def split_by_tag(row: dict) -> list[dict]:
    """One row per nonverbal tag type of row["transcript"], with "nv_tag"; uuid gets "_{tag}" if there are several."""
    tags = list(dict.fromkeys(m.group()[1:-1] for m in NV_TAG_RE.finditer(row["transcript"])))
    return [{**row, "uuid": f"{row['uuid']}_{tag}" if len(tags) > 1 else row["uuid"], "nv_tag": tag} for tag in tags]


def tag_runs(transcript: str, tag: str) -> list[tuple[int, int]]:
    """[start, end) of the runs of [tag]; instances with neither a word nor another tag between them form one run."""
    runs = []
    for m in re.finditer(re.escape(f"[{tag}]"), transcript):
        if runs:
            gap = transcript[runs[-1][1] : m.start()]
            if not CONTENT_CHAR_RE.search(gap) and not NV_TAG_RE.search(gap):
                runs[-1] = (runs[-1][0], m.end())
                continue
        runs.append((m.start(), m.end()))
    return runs


def count_phrase(words: list[str], phrase: list[str], lang: str) -> int:
    """Occurrences of the word sequence (en) or, overlapping, of the joined characters (zh); case-insensitive."""
    if lang == "zh":
        text, joined = "".join(words), "".join(phrase)
        return sum(text.startswith(joined, i) for i in range(len(text)))
    n = len(phrase)
    return sum(words[i : i + n] == phrase for i in range(len(words) - n + 1))


def side_anchor(
    transcript: str, tokens: list[tuple[str, int, int]], run: tuple[int, int], direction: str, tag: str, lang: str
) -> tuple[str, bool] | None:
    """Unique anchor words next to a tag run on one side ("before" = the words after it) and whether it touches the run.

    The anchor takes ANCHOR_UNITS without crossing punctuation (zh, once it meets punctuation while shorter than
    MIN_ZH_ANCHOR_CHARS, keeps crossing it), then grows until unique; it never crosses another instance of the tag.
    None if no such anchor exists.
    """
    seq = (
        [t for t in tokens if t[1] >= run[1]] if direction == "before" else [t for t in tokens if t[2] <= run[0]][::-1]
    )
    if not seq:
        return None
    if direction == "before":
        gaps = [transcript[edge : t[1]] for edge, t in zip([run[1]] + [t[2] for t in seq[:-1]], seq, strict=True)]
    else:
        gaps = [transcript[t[2] : edge] for edge, t in zip([run[0]] + [t[1] for t in seq[:-1]], seq, strict=True)]
    kinds = ["hard" if f"[{tag}]" in gap else "punct" if gap.strip() else "" for gap in gaps]
    if kinds[0] == "hard":
        return None

    def units(num_words: int) -> int:
        return num_words if lang == "en" else sum(len(word) for word, _, _ in seq[:num_words])

    def phrase(num_words: int) -> list[str]:
        words = [word for word, _, _ in seq[:num_words]]
        return words if direction == "before" else words[::-1]

    num_words, crossed = 1, False
    while num_words < len(seq) and units(num_words) < ANCHOR_UNITS[lang]:
        if kinds[num_words] == "hard":
            break
        if kinds[num_words] == "punct":
            if lang == "en" or (units(num_words) >= MIN_ZH_ANCHOR_CHARS and not crossed):
                break
            crossed = True
        num_words += 1
    if lang == "zh" and units(num_words) < MIN_ZH_ANCHOR_CHARS:
        return None
    words = [word.lower() for word, _, _ in tokens]
    while count_phrase(words, [word.lower() for word in phrase(num_words)], lang) != 1:
        if num_words == len(seq) or kinds[num_words] == "hard":
            return None
        num_words += 1
    if lang == "en" and num_words > MAX_EN_ANCHOR_WORDS:
        return None
    return (" " if lang == "en" else "").join(phrase(num_words)), kinds[0] == ""


def join_clauses(clauses: list[str], lang: str) -> str:
    """zh: joined with "，"; en: "Delete ..., delete ... ." with one capital and one final period."""
    if lang == "zh":
        return "，".join(clauses)
    text = ", ".join(clause.rstrip(".")[0].lower() + clause.rstrip(".")[1:] for clause in clauses)
    return text[0].upper() + text[1:] + "."


def make_candidate(row: dict, subset: str, rng: random.Random) -> dict | None:
    """Edit record deleting every [nv_tag] of the transcript; None if a tag sits inside a word or lacks an anchor."""
    transcript, tag, lang = row["transcript"], row["nv_tag"], row["language"]
    text, removed = strip_nv_tags(transcript, lang)
    if text != row["text"]:
        raise ValueError(f"{row['uuid']}: text is not the stripped transcript; run align with --use-transcript")
    aligned = tokenize_aligned(text, lang)
    offsets = [offset for removed_tag, offset in removed if removed_tag == tag]
    if any(start < offset < end for offset in offsets for _, start, end in aligned):
        return None

    tokens = []
    for piece in re.finditer(r"[^\[\]]+", NV_TAG_RE.sub(lambda m: "]" * len(m.group()), transcript)):
        if lang == "en":
            found = [(m.group(), m.start(), m.end()) for m in EN_ANCHOR_WORD_RE.finditer(piece.group())]
        else:
            found = tokenize_content(piece.group(), lang)
        tokens += [(word, piece.start() + s, piece.start() + e) for word, s, e in found]
    tag_text = f"[{tag}]"
    forwards, reverses = [], []
    for run in tag_runs(transcript, tag):
        sides = {d: side_anchor(transcript, tokens, run, d, tag, lang) for d in ("before", "after")}
        valid = [d for d in ("before", "after") if sides[d] is not None]
        if not valid:
            return None
        direction = next((d for d in valid if sides[d][1]), valid[0])
        anchor = sides[direction][0]
        forwards.append(ANCHORED_DELETION[direction][lang].format(x=anchor, y=tag_text))
        reverses.append(INSERTION[direction][lang].format(x=anchor, y=tag_text))

    edits = [
        {"op": "deletion", "target": tag_text, "new": "", "direction": "", "region": [m.start(), m.end()]}
        for m in re.finditer(re.escape(tag_text), transcript)
    ]
    record = build_record(
        row,
        "nonverbal",
        subset,
        "insertion",
        strip_nv_tags(transcript, lang, {tag})[0],
        join_clauses(forwards, lang),
        join_clauses(reverses, lang),
        edits,
        meta={"nv_tag": tag, "gaps": [sum(end <= offset for _, _, end in aligned) for offset in offsets]},
        original_text=transcript,
    )
    num_tokens = len(tokenize_content(text, lang))
    return {
        "row": row,
        "subset": subset,
        "length_bucket": length_bucket(num_tokens, SENTENCE_LENGTH_BOUNDS),
        "record": record,
    }


def complete_candidate(candidate: dict, llm: LLMClient | None, rng: random.Random) -> dict | None:
    return candidate["record"]
