import random

from voxcanvas.annotate.llm import LLMClient, complete_edits
from voxcanvas.annotate.records import MIN_KEEP_TOKENS, SENTENCE_LENGTH_BOUNDS, build_edits, build_record, length_bucket
from voxcanvas.annotate.templates import L_D, L_I, L_M
from voxcanvas.text import apply_edits, segment_ids, tokenize_content


SUBSETS = ("l_d", "l_m", "l_i")
NEEDS_LLM = frozenset({"l_m", "l_i"})
OPS = {"l_d": "deletion", "l_m": "substitution", "l_i": "insertion"}
TEMPLATES = {"l_d": L_D, "l_m": L_M, "l_i": L_I}
MAX_DELETE_TOKENS = 6  # longest l_d deletion, in tokens
DELETE_LENGTH_BOUNDS = tuple(range(1, MAX_DELETE_TOKENS + 1))


def make_candidate(row: dict, subset: str, rng: random.Random) -> dict:
    """Pick the utterance head or tail to edit (docs/pipeline.md §4.6)."""
    text, lang = row["text"], row["language"]
    tokens = tokenize_content(text, lang)
    position = rng.choice(("head", "tail"))
    num_tokens = 1
    if subset == "l_d":
        segments = segment_ids(text, tokens)
        edge_run = segments.count(segments[0] if position == "head" else segments[-1])
        num_tokens = rng.randint(1, min(edge_run, MAX_DELETE_TOKENS, len(tokens) - MIN_KEEP_TOKENS))
    picked = tokens[:num_tokens] if position == "head" else tokens[-num_tokens:]
    start, end = picked[0][1], picked[-1][2]
    direction = rng.choice(("before", "after")) if subset == "l_i" else ""
    (edit,) = build_edits(text, [[start, end]], OPS[subset], direction)
    return {
        "row": row,
        "subset": subset,
        "position": position,
        "edit": edit,
        "num_tokens": num_tokens,
        "length_bucket": (
            length_bucket(num_tokens, DELETE_LENGTH_BOUNDS)
            if subset == "l_d"
            else length_bucket(len(tokens), SENTENCE_LENGTH_BOUNDS)
        ),
    }


def complete_candidate(candidate: dict, llm: LLMClient | None, rng: random.Random) -> dict | None:
    """Apply the candidate's edit (LLM text for l_m / l_i) and render a random template; None if generation fails."""
    row, subset, position, edit = candidate["row"], candidate["subset"], candidate["position"], candidate["edit"]
    text, lang = row["text"], row["language"]
    if subset in NEEDS_LLM:
        assert llm is not None, f"{subset} needs an LLM client"
        completed = complete_edits(llm, text, [edit], edit["target"], lang)
        if completed is None:
            return None
        (edit,) = completed
    edited = apply_edits(text, [edit], lang)
    key = (position, edit["direction"]) if subset == "l_i" else position
    instruction = rng.choice(TEMPLATES[subset][key][lang]).format(x=edit["target"], y=edit["new"])
    meta = {"locator": "sentence_position", "position": position, "num_tokens": candidate["num_tokens"]}
    return build_record(row, "location_reference", subset, edit["op"], edited, instruction, None, [edit], meta)
