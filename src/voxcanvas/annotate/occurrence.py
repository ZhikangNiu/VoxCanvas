import random
from collections import defaultdict
from itertools import groupby, pairwise

from voxcanvas.annotate.llm import LLMClient, complete_edits
from voxcanvas.annotate.records import build_edits, build_record, length_bucket
from voxcanvas.annotate.templates import D1, D4, EN_ORDINALS, I1, I2, M1, M2, ordinal
from voxcanvas.text import apply_edits, content_key, count_occurrences, segment_ids, tokenize_content


SUBSETS = ("d1", "d4", "m1", "m2", "i1", "i2")
NEEDS_LLM = frozenset({"m1", "m2", "i1", "i2"})
# d4 / m2 / i2 edit the i-th occurrence; d1 / m1 / i1 edit all of them.
NTH_SUBSETS = frozenset({"d4", "m2", "i2"})
OPS = {
    "d1": "deletion",
    "d4": "deletion",
    "m1": "substitution",
    "m2": "substitution",
    "i1": "insertion",
    "i2": "insertion",
}
TEMPLATES = {"d1": D1, "d4": D4, "m1": M1, "m2": M2, "i1": I1, "i2": I2}
MAX_NTH_COUNT = len(EN_ORDINALS)  # one ordinal word per occurrence
# Lower bounds of the repeat-length buckets (in tokens): "1".."5", "6-7", "8-10", "11-15", "16+".
LENGTH_BOUNDS = (1, 2, 3, 4, 5, 6, 8, 11, 16)


def find_repeats(text: str, lang: str) -> list[dict]:
    """Maximal repeated token n-grams of `text`, as units sorted by first occurrence (docs/pipeline.md §4.5).

    A unit is {"text" (surface of the first occurrence), "norm", "count", "num_tokens", "occurrences"}; the
    occurrences are in text order and may overlap.
    """
    tokens = tokenize_content(text, lang)
    segments = [
        [(word.lower(), start, end) for _, (word, start, end) in group]
        for _, group in groupby(zip(segment_ids(text, tokens), tokens, strict=True), key=lambda pair: pair[0])
    ]
    places = defaultdict(list)
    for s, segment in enumerate(segments):
        norms = [norm for norm, _, _ in segment]
        for i in range(len(segment)):
            for j in range(i + 1, len(segment) + 1):
                places[tuple(norms[i:j])].append((s, i, j))
    units = []
    for key, found in places.items():
        if len(found) < 2:
            continue
        right = {segments[s][j][0] if j < len(segments[s]) else None for s, _, j in found}
        left = {segments[s][i - 1][0] if i > 0 else None for s, i, _ in found}
        if (len(right) == 1 and None not in right) or (len(left) == 1 and None not in left):
            continue
        spans = [[segments[s][i][1], segments[s][j - 1][2]] for s, i, j in found]
        units.append(
            {
                "text": text[spans[0][0] : spans[0][1]],
                "norm": ("" if lang == "zh" else " ").join(key),
                "count": len(found),
                "num_tokens": len(key),
                "occurrences": spans,
            }
        )
    return sorted(units, key=lambda unit: unit["occurrences"][0][0])


def make_candidate(row: dict, subset: str, rng: random.Random) -> dict | None:
    """Longest eligible repeat (ties: earliest), with the ordinal (d4/m2/i2) and insertion side (i1/i2) drawn.

    Eligible: non-overlapping occurrences and as many visible matches of the surface text as occurrences; for the
    ordinal subsets also at most MAX_NTH_COUNT occurrences; for d1, deleting all of them must leave some content.
    """
    text, lang = row["text"], row["language"]
    for unit in sorted(find_repeats(text, lang), key=lambda unit: (-unit["num_tokens"], unit["occurrences"][0][0])):
        spans = unit["occurrences"]
        if any(left[1] > right[0] for left, right in pairwise(spans)):
            continue
        if count_occurrences(text, unit["text"], lang) != unit["count"]:
            continue
        if subset in NTH_SUBSETS and unit["count"] > MAX_NTH_COUNT:
            continue
        if subset == "d1" and not content_key(apply_edits(text, build_edits(text, spans, "deletion"), lang)):
            continue
        return {
            "row": row,
            "subset": subset,
            "unit": unit,
            "ordinal": rng.randint(1, unit["count"]) if subset in NTH_SUBSETS else None,
            "direction": rng.choice(("before", "after")) if OPS[subset] == "insertion" else "",
            "length_bucket": length_bucket(unit["num_tokens"], LENGTH_BOUNDS),
        }
    return None


def complete_candidate(candidate: dict, llm: LLMClient | None, rng: random.Random) -> dict | None:
    """Edit all (or the i-th) occurrence(s); the LLM writes the replacement / insertion for m*/i*.

    An LLM answer is redrawn unless the edit changes the text, adds no new adjacent duplicate and is undone by the
    global reverse edit (replace every new text by the target / delete the new text next to every target).
    """
    row, subset, unit, nth = candidate["row"], candidate["subset"], candidate["unit"], candidate["ordinal"]
    raw, lang, direction = row["text"], row["language"], candidate["direction"]
    op = OPS[subset]
    spans = unit["occurrences"] if nth is None else [unit["occurrences"][nth - 1]]
    # Surface text of the edited occurrence(s); lowercased when they differ (e.g. "The" ... "the").
    surfaces = {raw[s:e] for s, e in spans}
    target = surfaces.pop() if len(surfaces) == 1 else raw[spans[0][0] : spans[0][1]].lower()

    edits = build_edits(raw, spans, op, direction)
    if subset in NEEDS_LLM:
        assert llm is not None, f"{subset} needs an LLM client"
        edits = complete_edits(llm, raw, edits, target, lang)
        if edits is None:
            return None
    new = edits[0]["new"]
    templates = TEMPLATES[subset][direction] if op == "insertion" else TEMPLATES[subset]
    instruction = rng.choice(templates[lang]).format(x=target, y=new, ord=ordinal(nth, lang) if nth else "")
    meta = {
        "locator": "all_occurrences" if nth is None else "nth_occurrence",
        "target": target,
        "count": unit["count"],
        "occurrences": unit["occurrences"],
    }
    if nth is not None:
        meta["ordinal"] = nth
    edited = apply_edits(raw, edits, lang)
    return build_record(row, "location_reference", subset, op, edited, instruction, None, edits, meta)
