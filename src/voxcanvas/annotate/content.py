import random

from voxcanvas.annotate.llm import LLMClient, generate_insertion, generate_replacement
from voxcanvas.annotate.prompts import INSERTION_PROMPT, SUBSTITUTION_PROMPT
from voxcanvas.annotate.records import MIN_KEEP_TOKENS, SENTENCE_LENGTH_BOUNDS, build_record, length_bucket
from voxcanvas.annotate.templates import (
    DELETION,
    INSERTION,
    INSTRUCTION_SEP,
    SUBSTITUTION,
    deletion_reverse,
    insertion_reverse,
)
from voxcanvas.text import (
    apply_edits,
    content_key,
    count_occurrences,
    edit_points,
    edit_round_trips,
    find_occurrences,
    has_adjacent_duplicate,
    segment_ids,
    tokenize_content,
    undo_insertion,
)


SUBSETS = ("deletion", "insertion", "substitution", "multi")
NEEDS_LLM = frozenset({"insertion", "substitution", "multi"})
MAX_SPAN_TOKENS = 6  # sampled deletion / substitution length, before uniqueness growth
MAX_SPAN_FRACTION = 0.6  # longest edited span, as a fraction of the sentence characters
MAX_LENGTH_DELTA = 20  # words (en) / characters (zh) new text may add over its target
SUBSTITUTION_LENGTH_SCALE = (0.5, 3.0)  # hinted new-text length, as a multiple of the target length
INSERTION_LENGTH_HINT = (1, 20)  # inclusive range of the hinted inserted length, words (en) / characters (zh)
NUM_MULTI_EDITS = (2, 6)  # inclusive range of the targeted number of multi-span edits
PLACEMENT_TRIES = 8  # multi-span placement attempts per targeted edit
# Multi-span sub-edit kinds (op, insertion direction) and their sampling weights.
MULTI_KINDS = (("deletion", ""), ("substitution", ""), ("insertion", "before"), ("insertion", "after"))
MULTI_WEIGHTS = (2, 2, 1, 1)


def length_units(text: str, lang: str) -> int:
    """Length in the units of the LLM length hint: words (en) or content characters (zh)."""
    return len(tokenize_content(text, lang)) if lang == "en" else len(content_key(text))


def select_span(
    text: str, tokens: list[tuple[str, int, int]], idx: int, num_tokens: int, lang: str
) -> tuple[int, int] | None:
    """Token range [lo, hi) of a deletion / substitution span from token idx that occurs once in text.

    The span grows until it is unique and is retried shorter when it exceeds MAX_SPAN_FRACTION of the text
    (docs/pipeline.md §4.3); None if even a single token cannot become unique.
    """
    for span_tokens in range(num_tokens, 0, -1):
        lo, hi, prefer_left = idx, idx + span_tokens, True
        while True:
            start, end = tokens[lo][1], tokens[hi - 1][2]
            if count_occurrences(text, text[start:end], lang) == 1:
                if end - start <= MAX_SPAN_FRACTION * len(text):
                    return lo, hi
                break
            if lo > 0 and (prefer_left or hi == len(tokens)):
                lo -= 1
            elif hi < len(tokens):
                hi += 1
            else:
                break
            prefer_left = not prefer_left
    return None


def unique_anchor(
    text: str, tokens: list[tuple[str, int, int]], segments: list[int], idx: int, direction: str, lang: str
) -> tuple[int, int] | None:
    """Token range [lo, hi) of a unique insertion anchor that keeps token idx's edge at the insertion point.

    The anchor starts as token idx and grows away from the insertion point (left for "after", right for
    "before") inside its punctuation segment; None if it never becomes unique.
    """
    lo = hi = idx
    while count_occurrences(text, text[tokens[lo][1] : tokens[hi][2]], lang) != 1:
        if direction == "after" and lo > 0 and segments[lo - 1] == segments[idx]:
            lo -= 1
        elif direction == "before" and hi + 1 < len(tokens) and segments[hi + 1] == segments[idx]:
            hi += 1
        else:
            return None
    return lo, hi + 1


def place_edit(
    text: str,
    tokens: list[tuple[str, int, int]],
    segments: list[int],
    idx: int,
    op: str,
    direction: str,
    max_tokens: int,
    lang: str,
    rng: random.Random,
) -> tuple[dict, int, int] | None:
    """Edit of `op` at token idx (new text still empty) and the token range [lo, hi) it occupies.

    Deletions / substitutions draw a length of 1..max_tokens tokens; an insertion's target is its anchor and its
    region is the selected token.
    """
    if op == "insertion":
        found = unique_anchor(text, tokens, segments, idx, direction, lang)
    else:
        found = select_span(text, tokens, idx, rng.randint(1, max_tokens), lang)
    if found is None:
        return None
    lo, hi = found
    start, end = tokens[lo][1], tokens[hi - 1][2]
    region = [tokens[idx][1], tokens[idx][2]] if op == "insertion" else [start, end]
    return {"op": op, "target": text[start:end], "new": "", "direction": direction, "region": region}, lo, hi


def make_candidate(row: dict, subset: str, rng: random.Random) -> dict | None:
    """Rule-based edit positions: one edit for single_span subsets, 2-6 separated edits for "multi"."""
    text, lang = row["text"], row["language"]
    tokens = tokenize_content(text, lang)
    segments = segment_ids(text, tokens)
    num_tokens = len(tokens)
    candidate = {"row": row, "subset": subset, "length_bucket": length_bucket(num_tokens, SENTENCE_LENGTH_BOUNDS)}
    if subset != "multi":
        idx = rng.randrange(num_tokens)
        direction = rng.choice(("before", "after")) if subset == "insertion" else ""
        max_tokens = min(MAX_SPAN_TOKENS, num_tokens - idx)
        if subset == "deletion":
            max_tokens = min(max_tokens, num_tokens - MIN_KEEP_TOKENS)
        placed = place_edit(text, tokens, segments, idx, subset, direction, max_tokens, lang, rng)
        if placed is None:
            return None
        edit, _, _ = placed
        return {**candidate, "edits": [edit]}

    num_target = rng.randint(*NUM_MULTI_EDITS)
    used = [False] * num_tokens
    edits = []
    for _ in range(num_target * PLACEMENT_TRIES):
        if len(edits) == num_target:
            break
        idx = rng.randrange(num_tokens)
        if used[idx]:
            continue
        op, direction = rng.choices(MULTI_KINDS, weights=MULTI_WEIGHTS)[0]
        max_tokens = min(MAX_SPAN_TOKENS, num_tokens - idx)
        placed = place_edit(text, tokens, segments, idx, op, direction, max_tokens, lang, rng)
        if placed is None:
            continue
        edit, lo, hi = placed
        # Each edit also reserves one buffer token per side, so two edits are at least two tokens apart.
        reserved = range(max(lo - 1, 0), min(hi + 1, num_tokens))
        if any(used[i] for i in reserved):
            continue
        for i in reserved:
            used[i] = True
        edits.append(edit)
    if len(edits) < NUM_MULTI_EDITS[0]:
        return None
    edits.sort(key=lambda edit: edit["region"][0])
    return {**candidate, "edits": edits, "num_edits_target": num_target}


def generate_new(llm: LLMClient, raw: str, edit: dict, lang: str, rng: random.Random, check_edit: bool) -> str | None:
    """LLM replacement / insertion text for `edit` with a sampled length hint; None if every redraw fails.

    Answers are redrawn when they add more than MAX_LENGTH_DELTA units and, with check_edit (single_span), when the
    applied edit does not change the text, adds an adjacent duplicate, or cannot be reversed unambiguously.
    """
    target, direction = edit["target"], edit["direction"]
    substitution = edit["op"] == "substitution"
    base_length = length_units(target, lang) if substitution else 0

    def accept(new: str) -> bool:
        if length_units(new, lang) - base_length > MAX_LENGTH_DELTA:
            return False
        if not check_edit:
            return True
        edited = apply_edits(raw, [{**edit, "new": new}], lang)
        if not substitution and insertion_reverse(edited, target, new, direction, lang) is None:
            return False
        return edit_round_trips(raw, edited, target, new, edit["op"], direction, lang)

    if substitution:
        target_length = max(1, round(base_length * rng.uniform(*SUBSTITUTION_LENGTH_SCALE)))
        return generate_replacement(llm, raw, target, lang, SUBSTITUTION_PROMPT, accept, target_length)
    return generate_insertion(
        llm, raw, target, direction, lang, INSERTION_PROMPT, accept, rng.randint(*INSERTION_LENGTH_HINT)
    )


def replay_reverse(edited: str, edits: list[dict], contexts: list[tuple[str, str] | None], lang: str) -> str | None:
    """Apply the reverse of every edit to `edited`, right to left; None if a deletion context is missing.

    Substitutions are swapped back globally, insertions removed next to their anchor, and deleted text is put
    back before / after its (context, relation).
    """
    text = edited
    for edit, context in reversed(list(zip(edits, contexts, strict=True))):
        if edit["op"] == "substitution":
            text = text.replace(edit["new"], edit["target"])
        elif edit["op"] == "insertion":
            text = undo_insertion(text, edit["target"], edit["new"], edit["direction"], lang)
        else:
            found = find_occurrences(text, context[0], lang)
            if not found:
                return None
            restore = {"op": "insertion", "new": edit["target"], "direction": context[1], "region": found[0]}
            text = apply_edits(text, [restore], lang)
    return text


def complete_candidate(candidate: dict, llm: LLMClient | None, rng: random.Random) -> dict | None:
    """Generate new text, apply the edits, build forward / reverse instructions and check reversibility."""
    row, subset = candidate["row"], candidate["subset"]
    raw, lang = row["text"], row["language"]
    single = subset != "multi"
    edits = []
    for edit in candidate["edits"]:
        if edit["op"] != "deletion":
            assert llm is not None, f"{subset} needs an LLM client"
            new = generate_new(llm, raw, edit, lang, rng, check_edit=single)
            if new is None:
                return None
            edit = {**edit, "new": new}
        edits.append(edit)
    edited = apply_edits(raw, edits, lang)
    if not content_key(edited) or content_key(edited) == content_key(raw):
        return None
    if not single and has_adjacent_duplicate(edited, lang) and not has_adjacent_duplicate(raw, lang):
        return None
    # A substitution's reverse names its new text alone, unlike an insertion's, which can fall back to its anchor.
    if not single and any(
        edit["op"] == "substitution" and count_occurrences(edited, edit["new"], lang) != 1 for edit in edits
    ):
        return None

    forwards, reverses, contexts = [], [], []
    for edit, point in zip(edits, edit_points(raw, edited, edits), strict=True):
        target, new, direction = edit["target"], edit["new"], edit["direction"]
        context = None
        if edit["op"] == "deletion":
            found = deletion_reverse(edited, point, target, lang, rng, raw if single else None)
            if found is None:
                return None
            reverse, anchor, relation = found
            forward = DELETION[lang].format(x=target)
            context = (anchor, relation)
        elif edit["op"] == "substitution":
            forward = SUBSTITUTION[lang].format(x=target, y=new)
            reverse = SUBSTITUTION[lang].format(x=new, y=target)
        else:
            forward = INSERTION[direction][lang].format(x=target, y=new)
            reverse = insertion_reverse(edited, target, new, direction, lang, None if single else rng)
            if reverse is None:
                return None
        forwards.append(forward)
        reverses.append(reverse)
        contexts.append(context)

    if single:
        return build_record(row, "single_span", subset, subset, edited, forwards[0], reverses[0], edits)
    replay = replay_reverse(edited, edits, contexts, lang)
    if replay is None or content_key(replay) != content_key(raw):
        return None
    rng.shuffle(reverses)
    sep = INSTRUCTION_SEP[lang]
    meta = {"num_edits_target": candidate["num_edits_target"]}
    return build_record(
        row, "multi_span", "multi", "multi", edited, sep.join(forwards), sep.join(reverses), edits, meta
    )
