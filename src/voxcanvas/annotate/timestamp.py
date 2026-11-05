import random

from voxcanvas.annotate.llm import LLMClient, complete_edits
from voxcanvas.annotate.records import MIN_KEEP_TOKENS, SENTENCE_LENGTH_BOUNDS, build_edits, build_record, length_bucket
from voxcanvas.annotate.templates import L_T_D, L_T_I, L_T_M, format_time
from voxcanvas.text import apply_edits, segment_ids, tokenize_aligned, tokenize_content


SUBSETS = ("l_t_d", "l_t_m", "l_t_i")
NEEDS_LLM = frozenset({"l_t_m", "l_t_i"})
OPS = {"l_t_d": "deletion", "l_t_m": "substitution", "l_t_i": "insertion"}
TEMPLATES = {"l_t_d": L_T_D, "l_t_m": L_T_M, "l_t_i": L_T_I}
MAX_LOCATED_TOKENS = 5  # most words a span-mode time locator covers
DELETE_LENGTH_BOUNDS = tuple(range(1, MAX_LOCATED_TOKENS + 1))


def word_time_spans(
    text: str, tokens: list[tuple[str, int, int]], align_word: list[list], lang: str
) -> list[tuple[float, float]]:
    """(start, end) seconds of each token, from the first to the last align token it overlaps.

    `align_word` must match tokenize_aligned(text) (text.gate).
    """
    aligned = tokenize_aligned(text, lang)
    spans = []
    for _, start, end in tokens:
        covered = [k for k, (_, s, e) in enumerate(aligned) if s < end and e > start]
        spans.append((align_word[covered[0]][1], align_word[covered[-1]][2]))
    return spans


def words_at_shown_time(spans: list[tuple[float, float]], mode: str, t_start: float, t_end: float) -> list[int]:
    """Indices of the words that the times as shown in the instruction (format_time) refer to.

    point: the words whose span contains the shown time, else the nearest word(s); span: the words whose midpoint
    lies inside the shown interval.
    """
    shown_start, shown_end = float(format_time(t_start)), float(format_time(t_end))
    # Round away float noise so midpoints on a shown boundary and tied gaps compare exactly.
    if mode == "span":
        return [i for i, (s, e) in enumerate(spans) if shown_start <= round((s + e) / 2, 6) <= shown_end]
    inside = [i for i, (s, e) in enumerate(spans) if s <= shown_start <= e]
    if inside:
        return inside
    gaps = [round(min(abs(shown_start - s), abs(shown_start - e)), 6) for s, e in spans]
    return [i for i, gap in enumerate(gaps) if gap == min(gaps)]


def make_candidate(row: dict, subset: str, rng: random.Random) -> dict | None:
    """Locate one word by its midpoint (point) or 1-5 words of one punctuation segment by their start/end (span).

    l_t_i is point only. None when the rounded time(s) do not single out exactly the located words, or when an l_t_d
    deletion would keep fewer than two tokens.
    """
    text, lang = row["text"], row["language"]
    tokens = tokenize_content(text, lang)
    spans = word_time_spans(text, tokens, row["align_word"], lang)
    mode = "point" if subset == "l_t_i" else rng.choice(("point", "span"))
    has_word = rng.random() < 0.5
    lo = rng.randrange(len(tokens))
    if mode == "point":
        hi = lo + 1
        t_start = t_end = (spans[lo][0] + spans[lo][1]) / 2
    else:
        segments = segment_ids(text, tokens)
        hi = lo + rng.randint(1, min(MAX_LOCATED_TOKENS, segments[lo:].count(segments[lo])))
        t_start, t_end = spans[lo][0], spans[hi - 1][1]
    if subset == "l_t_d" and len(tokens) - (hi - lo) < MIN_KEEP_TOKENS:
        return None
    if words_at_shown_time(spans, mode, t_start, t_end) != list(range(lo, hi)):
        return None
    start, end = tokens[lo][1], tokens[hi - 1][2]
    direction = rng.choice(("before", "after")) if subset == "l_t_i" else ""
    (edit,) = build_edits(text, [[start, end]], OPS[subset], direction)
    return {
        "row": row,
        "subset": subset,
        "mode": mode,
        "has_word": has_word,
        "token_range": [lo, hi],
        "edit": edit,
        "t_start": t_start,
        "t_end": t_end,
        "length_bucket": (
            length_bucket(hi - lo, DELETE_LENGTH_BOUNDS)
            if subset == "l_t_d"
            else length_bucket(len(tokens), SENTENCE_LENGTH_BOUNDS)
        ),
    }


def complete_candidate(candidate: dict, llm: LLMClient | None, rng: random.Random) -> dict | None:
    """Apply the candidate's edit (LLM text for l_t_m / l_t_i) and render a random time template."""
    row, subset, mode, has_word = candidate["row"], candidate["subset"], candidate["mode"], candidate["has_word"]
    text, lang, edit = row["text"], row["language"], candidate["edit"]
    if subset in NEEDS_LLM:
        assert llm is not None, f"{subset} needs an LLM client"
        completed = complete_edits(llm, text, [edit], edit["target"], lang)
        if completed is None:
            return None
        (edit,) = completed
    edited = apply_edits(text, [edit], lang)
    key = (edit["direction"], has_word) if subset == "l_t_i" else (mode, has_word)
    shown_start, shown_end = format_time(candidate["t_start"]), format_time(candidate["t_end"])
    instruction = rng.choice(TEMPLATES[subset][key][lang]).format(
        t=shown_start, ts=shown_start, te=shown_end, x=edit["target"], y=edit["new"]
    )
    meta = {
        "locator": "timestamp",
        "mode": mode,
        "has_word": has_word,
        "t_start": round(candidate["t_start"], 3),
        "t_end": round(candidate["t_end"], 3),
        "token_range": candidate["token_range"],
    }
    return build_record(row, "location_reference", subset, edit["op"], edited, instruction, None, [edit], meta)
