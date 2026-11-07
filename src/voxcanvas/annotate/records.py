from voxcanvas.text import EDIT_OPS, edit_keeps_alignment
from voxcanvas.utils import rng_for


SUBSETS_OF_TASK = {
    "single_span": ("deletion", "insertion", "substitution"),
    "multi_span": ("multi",),
    "location_reference": ("d1", "d4", "m1", "m2", "i1", "i2", "l_d", "l_m", "l_i", "l_t_d", "l_t_m", "l_t_i"),
    "nonverbal": ("insertion",),
}
EDIT_TYPES = (*EDIT_OPS, "multi")
SENTENCE_LENGTH_BOUNDS = (4, 8, 16)  # sentence content tokens: "4-7", "8-15", "16+"
MIN_KEEP_TOKENS = 2  # content tokens a sampled deletion must leave


def record_id(task: str, subset: str, uuid: str) -> str:
    return f"{task}/{subset}/{uuid}"


def build_edits(text: str, spans: list[list[int]], op: str, direction: str = "") -> list[dict]:
    """One edit per [start, end) span of `text`, with its new text still empty."""
    return [{"op": op, "target": text[s:e], "new": "", "direction": direction, "region": [s, e]} for s, e in spans]


def build_record(
    row: dict,
    task: str,
    subset: str,
    edit_type: str,
    edited_text: str,
    instruction: str,
    reverse_instruction: str | None,
    edits: list[dict],
    meta: dict | None = None,
    original_text: str | None = None,
) -> dict | None:
    """Edit record (docs/pipeline.md §8.3) for an aligned row; `edits` are sorted by region start.

    None when the edits re-tokenize the text outside their regions (`text.edit_keeps_alignment`), which would move
    the synthesized span onto words the instruction does not name. `original_text` (nonverbal: the tagged transcript
    the edits refer to) replaces row["text"] and skips that check, as its edits remove tags, never words.
    """
    assert subset in SUBSETS_OF_TASK[task] and edit_type in EDIT_TYPES, (task, subset, edit_type)
    assert all(edit["op"] in EDIT_OPS for edit in edits), edits
    assert (reverse_instruction is None) == (task == "location_reference"), (task, reverse_instruction)
    if original_text is None and not edit_keeps_alignment(row["text"], edited_text, edits, row["language"]):
        return None
    return {
        "id": record_id(task, subset, row["uuid"]),
        "task": task,
        "subset": subset,
        "edit_type": edit_type,
        "language": row["language"],
        "uuid": row["uuid"],
        "audio_path": row["audio_path"],
        "align_word": row["align_word"],
        "original_text": row["text"] if original_text is None else original_text,
        "edited_text": edited_text,
        "instruction": instruction,
        "reverse_instruction": reverse_instruction,
        "edits": sorted(edits, key=lambda edit: edit["region"][0]),
        "num_edits": len(edits),
        "meta": meta or {},
    }


def length_bucket(num_units: int, lower_bounds: tuple[int, ...]) -> str:
    """Label of the bucket holding num_units, e.g. lower_bounds (4, 8, 16) gives "4-7", "8-15", "16+"."""
    i = max(sum(bound <= num_units for bound in lower_bounds) - 1, 0)
    low = lower_bounds[i]
    if i == len(lower_bounds) - 1:
        return f"{low}+"
    high = lower_bounds[i + 1] - 1
    return str(low) if low == high else f"{low}-{high}"


def stratified_sample(candidates: list[dict], num_samples: int, seed: int, subset: str) -> list[dict]:
    """Pick `num_samples` candidates balanced over candidate["length_bucket"], kept in input order.

    Buckets are visited round-robin from the longest (largest lower bound), each in order of a random priority per
    (seed, uuid, subset), so the picks depend on the candidate set, not its order. num_samples <= 0 keeps everything.
    """
    if num_samples <= 0 or num_samples >= len(candidates):
        return list(candidates)
    by_bucket = {}
    for i, candidate in enumerate(candidates):
        by_bucket.setdefault(candidate["length_bucket"], []).append(i)
    order = sorted(by_bucket, key=lambda label: int(label.split("-")[0].rstrip("+")), reverse=True)
    for label in order:
        by_bucket[label].sort(key=lambda i: rng_for(seed, candidates[i]["row"]["uuid"], subset, "sample").random())
    picked = []
    while len(picked) < num_samples:
        for label in order:
            if by_bucket[label] and len(picked) < num_samples:
                picked.append(by_bucket[label].pop())
    return [candidates[i] for i in sorted(picked)]
