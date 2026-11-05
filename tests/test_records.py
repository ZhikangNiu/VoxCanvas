from collections import Counter

import pytest

from voxcanvas import text
from voxcanvas.annotate import boundary, content, llm, occurrence, timestamp
from voxcanvas.annotate.records import (
    MIN_KEEP_TOKENS,
    SENTENCE_LENGTH_BOUNDS,
    build_record,
    length_bucket,
    record_id,
    stratified_sample,
)


ROW = {
    "uuid": "EN_B00000_S00000_W000000",
    "audio_path": "examples/mini/audio/x.mp3",
    "language": "en",
    "text": "I like the red car.",
    "align_word": [["I", 0.0, 0.1]],
    "duration": 1.0,
}


def test_release_parameters() -> None:
    """These values must stay equal to the released pipeline (docs/pipeline.md §4.1-§4.7)."""
    assert (
        content.MAX_SPAN_TOKENS,
        content.MAX_SPAN_FRACTION,
        content.MAX_LENGTH_DELTA,
        content.SUBSTITUTION_LENGTH_SCALE,
        content.INSERTION_LENGTH_HINT,
        content.NUM_MULTI_EDITS,
        content.PLACEMENT_TRIES,
        content.MULTI_WEIGHTS,
    ) == (6, 0.6, 20, (0.5, 3.0), (1, 20), (2, 6), 8, (2, 2, 1, 1))
    assert (text.MIN_CONTENT_TOKENS, MIN_KEEP_TOKENS, SENTENCE_LENGTH_BOUNDS, llm.MAX_DRAWS) == (4, 2, (4, 8, 16), 5)
    assert (boundary.MAX_DELETE_TOKENS, occurrence.MAX_NTH_COUNT, timestamp.MAX_LOCATED_TOKENS) == (6, 8, 5)


def test_build_record_fields_and_sorted_edits() -> None:
    edits = [
        {"op": "substitution", "target": "car", "new": "bike", "direction": "", "region": [15, 18]},
        {"op": "deletion", "target": "red", "new": "", "direction": "", "region": [11, 14]},
    ]
    record = build_record(ROW, "multi_span", "multi", "multi", "I like the bike.", "fwd", "rev", edits, {"k": 2})
    assert record["id"] == "multi_span/multi/EN_B00000_S00000_W000000" == record_id("multi_span", "multi", ROW["uuid"])
    assert record["original_text"] == ROW["text"] and record["audio_path"] == ROW["audio_path"]
    assert [edit["region"][0] for edit in record["edits"]] == [11, 15]
    assert record["meta"] == {"k": 2} and "duration" not in record
    location = build_record(ROW, "location_reference", "d4", "deletion", "I like the car.", "fwd", None, edits[1:])
    assert location["reverse_instruction"] is None and location["meta"] == {}
    with pytest.raises(AssertionError):
        build_record(ROW, "single_span", "deletion", "delete", "x", "fwd", "rev", edits[1:])
    with pytest.raises(AssertionError):
        build_record(ROW, "single_span", "d4", "deletion", "x", "fwd", "rev", edits[1:])


def test_build_record_drops_edits_that_re_tokenize_the_text() -> None:
    row = {**ROW, "language": "zh", "text": "日本企业总共21起500亿日元以上的并购案中，有18起是美国公司。"}
    edits = [{"op": "deletion", "target": "起", "new": "", "direction": "", "region": [8, 9]}]
    edited = text.apply_edits(row["text"], edits, "zh")
    assert edited.startswith("日本企业总共21500亿")
    assert build_record(row, "location_reference", "d4", "deletion", edited, "fwd", None, edits) is None


def test_length_bucket() -> None:
    assert [length_bucket(num_units, (4, 8, 16)) for num_units in (2, 4, 7, 8, 15, 16, 40)] == [
        "4-7",
        "4-7",
        "4-7",
        "8-15",
        "8-15",
        "16+",
        "16+",
    ]
    buckets = [length_bucket(num_units, (1, 2, 3, 6, 8)) for num_units in (1, 2, 5, 7, 9)]
    assert buckets == ["1", "2", "3-5", "6-7", "8+"]


def test_stratified_sample_balances_buckets_and_keeps_input_order() -> None:
    buckets = ["4-7"] * 50 + ["16+"] * 3 + ["8-15"] * 50
    candidates = [{"row": {"uuid": f"u{i:03d}"}, "length_bucket": bucket} for i, bucket in enumerate(buckets)]
    picked = stratified_sample(candidates, 21, 0, "deletion")
    assert len(picked) == 21
    uuids = [c["row"]["uuid"] for c in picked]
    assert uuids == sorted(uuids)
    assert Counter(c["length_bucket"] for c in picked) == {"16+": 3, "8-15": 9, "4-7": 9}
    assert stratified_sample(candidates, 0, 0, "deletion") == candidates
    # The picks depend on the candidate set, not on its order (e.g. a re-merged sharded input).
    assert stratified_sample(candidates[::-1], 21, 0, "deletion") == picked[::-1]
    assert stratified_sample(candidates, 21, 0, "insertion") != picked
    # Buckets are visited from the longest, so a single pick comes from "16+".
    (pick,) = stratified_sample(candidates, 1, 0, "deletion")
    assert pick["length_bucket"] == "16+"
