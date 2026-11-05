import random
from collections import Counter

import pytest
from conftest import ScriptedLLM, candidate_with, make_row

from voxcanvas.annotate import timestamp
from voxcanvas.annotate.prompts import GLOBAL_SUBSTITUTION_PROMPT
from voxcanvas.annotate.templates import L_T_D, L_T_I, L_T_M, format_time
from voxcanvas.text import CONTENT_CHAR_RE, tokenize_content
from voxcanvas.utils import rng_for


def candidates(row: dict, subset: str, num_seeds: int = 300) -> list[dict]:
    found = [timestamp.make_candidate(row, subset, random.Random(seed)) for seed in range(num_seeds)]
    return [candidate for candidate in found if candidate is not None]


def spans_of(row: dict) -> list[tuple[float, float]]:
    tokens = tokenize_content(row["text"], row["language"])
    return timestamp.word_time_spans(row["text"], tokens, row["align_word"], row["language"])


def shown_words(spans: list[tuple[float, float]], candidate: dict) -> list[int]:
    return timestamp.words_at_shown_time(spans, candidate["mode"], candidate["t_start"], candidate["t_end"])


def test_word_time_spans() -> None:
    row = make_row("基本上他们每天都要出去玩耍。", "zh")
    align = row["align_word"]
    spans = spans_of(row)
    assert spans[0] == (align[0][1], align[2][2])
    assert len(spans) == len(tokenize_content(row["text"], "zh"))
    row = make_row("a well-known fact here", "en")
    align = row["align_word"]
    spans = spans_of(row)
    assert spans == [tuple(word[1:]) for word in align]


def test_point_is_word_midpoint_and_span_is_word_bounds() -> None:
    row = make_row("今天天气非常好我们出去玩吧。", "zh")
    spans = spans_of(row)
    found = candidates(row, "l_t_d")
    assert {candidate["mode"] for candidate in found} == {"point", "span"}
    for candidate in found:
        lo, hi = candidate["token_range"]
        if candidate["mode"] == "point":
            assert hi == lo + 1
            assert candidate["t_start"] == candidate["t_end"] == pytest.approx(sum(spans[lo]) / 2)
        else:
            assert (candidate["t_start"], candidate["t_end"]) == (spans[lo][0], spans[hi - 1][1])
        assert shown_words(spans, candidate) == list(range(lo, hi))


def test_span_stays_in_one_segment_and_has_at_most_five_words() -> None:
    row = make_row("we really went to the old market in town, then we all went back home", "en")
    lengths = Counter()
    for candidate in candidates(row, "l_t_m", 500):
        start, end = candidate["edit"]["region"]
        assert "," not in row["text"][start:end]
        lengths[candidate["token_range"][1] - candidate["token_range"][0]] += 1
    assert set(lengths) == {1, 2, 3, 4, 5}


def test_l_t_d_keeps_two_words_and_buckets_by_deleted_words() -> None:
    row = make_row("we went home today", "en")
    for candidate in candidates(row, "l_t_d"):
        num_deleted = candidate["token_range"][1] - candidate["token_range"][0]
        assert num_deleted <= 2
        assert candidate["length_bucket"] == str(num_deleted)
    assert {candidate["length_bucket"] for candidate in candidates(row, "l_t_m")} == {"4-7"}


def test_words_at_shown_time() -> None:
    # point: inside one word, on a shared boundary (ambiguous), in a pause (nearest word), equidistant (ambiguous)
    assert timestamp.words_at_shown_time([(1.0, 1.2), (1.2, 1.4)], "point", 1.14, 1.14) == [0]
    assert timestamp.words_at_shown_time([(1.0, 1.2), (1.2, 1.26)], "point", 1.23, 1.23) == [0, 1]
    assert timestamp.words_at_shown_time([(0.5, 1.1), (1.26, 1.28)], "point", 1.27, 1.27) == [1]
    assert timestamp.words_at_shown_time([(0.5, 1.15), (1.25, 1.27)], "point", 1.2, 1.2) == [0, 1]
    # span: word midpoints inside the shown interval
    spans = [(0.0, 0.5), (0.5, 0.62), (0.62, 1.0)]
    assert timestamp.words_at_shown_time(spans, "span", 0.5, 0.62) == [1]
    assert timestamp.words_at_shown_time([(0.40, 0.44), (0.44, 0.8)], "span", 0.44, 0.8) == [0, 1]
    assert timestamp.words_at_shown_time([(0.0, 0.1), (0.1, 0.14), (0.14, 0.6)], "span", 0.1, 0.14) == []


def test_rounding_ambiguity_drops_candidates() -> None:
    # Real Emilia timing: zero-length "a" at 2.24 s is shown as 2.2 s, which lies inside "leave".
    times = [(1.84, 2.08), (2.08, 2.24), (2.24, 2.24), (2.24, 2.56), (2.56, 3.04), (3.04, 3.2), (3.2, 3.44)]
    row = make_row("didn't leave a bad situation back home", "en", times)
    spans = spans_of(row)
    assert timestamp.words_at_shown_time(spans, "point", 2.24, 2.24) == [1]
    found = candidates(row, "l_t_d", 500)
    assert len(found) < 500
    assert {"point", "span"} == {candidate["mode"] for candidate in found}
    for candidate in found:
        lo, hi = candidate["token_range"]
        assert candidate["token_range"] != [2, 3], candidate
        assert shown_words(spans, candidate) == list(range(lo, hi))


def test_l_t_d_record() -> None:
    row = make_row("we went to the old market today", "en")
    candidate = candidate_with(timestamp.make_candidate, row, "l_t_d", mode="span", token_range=[3, 5], has_word=True)
    record = timestamp.complete_candidate(candidate, None, random.Random(0))
    assert record["id"] == "location_reference/l_t_d/u1"
    assert record["edited_text"] == "we went to market today"
    assert record["instruction"] in [t.format(ts="0.6", te="0.9", x="the old") for t in L_T_D[("span", True)]["en"]]
    assert record["reverse_instruction"] is None
    assert record["edits"] == [{"op": "deletion", "target": "the old", "new": "", "direction": "", "region": [11, 18]}]
    assert record["meta"] == {
        "locator": "timestamp",
        "mode": "span",
        "has_word": True,
        "t_start": 0.6,
        "t_end": 0.9,
        "token_range": [3, 5],
    }
    candidate = candidate_with(timestamp.make_candidate, row, "l_t_d", mode="point", token_range=[0, 1], has_word=False)
    record = timestamp.complete_candidate(candidate, None, random.Random(0))
    assert record["edited_text"] == "went to the old market today"
    assert record["instruction"] in [t.format(t="0.1") for t in L_T_D[("point", False)]["en"]]


def test_l_t_m_replaces_located_words_with_one_phrase() -> None:
    row = make_row("我们今天下午去公园散步。", "zh")
    candidate = candidate_with(timestamp.make_candidate, row, "l_t_m", mode="span", has_word=True)
    start, end = candidate["edit"]["region"]
    target = row["text"][start:end]
    llm = ScriptedLLM(["散步", "明天早上"])
    record = timestamp.complete_candidate(candidate, llm, random.Random(0))
    assert record["edited_text"] == row["text"][:start] + "明天早上" + row["text"][end:]
    assert record["edit_type"] == "substitution"
    shown = format_time(candidate["t_start"]), format_time(candidate["t_end"])
    assert record["instruction"] in [
        t.format(ts=shown[0], te=shown[1], x=target, y="明天早上") for t in L_T_M[("span", True)]["zh"]
    ]
    assert llm.prompts[0] == GLOBAL_SUBSTITUTION_PROMPT.format(raw_text=row["text"], target=target)


def test_l_t_i_is_point_only_and_rejects_word_already_next_to_anchor() -> None:
    row = make_row("Let's call her tonight, okay?", "en")
    assert {candidate["mode"] for candidate in candidates(row, "l_t_i")} == {"point"}
    candidate = candidate_with(
        timestamp.make_candidate, row, "l_t_i", token_range=[0, 1], direction="after", has_word=True
    )
    record = timestamp.complete_candidate(candidate, ScriptedLLM(["call", "quickly"]), random.Random(0))
    assert record["edited_text"] == "Let's quickly call her tonight, okay?"
    assert record["edits"][0] == {
        "op": "insertion",
        "target": "Let's",
        "new": "quickly",
        "direction": "after",
        "region": [0, 5],
    }
    assert record["instruction"] in [t.format(t="0.1", x="Let's", y="quickly") for t in L_T_I[("after", True)]["en"]]
    assert record["meta"]["mode"] == "point" and record["meta"]["token_range"] == [0, 1]


def test_template_choice_is_independent_of_mode() -> None:
    row = make_row("we went to the old market today", "en")
    combos = Counter()
    for i in range(300):
        uuid = f"u{i}"
        candidate = timestamp.make_candidate(row, "l_t_m", rng_for(0, uuid, "l_t_m", "candidate"))
        record = timestamp.complete_candidate(candidate, ScriptedLLM(["zebra"]), rng_for(0, uuid, "l_t_m", "complete"))
        key = (candidate["mode"], candidate["has_word"])
        start, end = candidate["edit"]["region"]
        shown = format_time(candidate["t_start"]), format_time(candidate["t_end"])
        templates = [
            t.format(t=shown[0], ts=shown[0], te=shown[1], x=row["text"][start:end], y="zebra")
            for t in L_T_M[key]["en"]
        ]
        combos[(candidate["mode"], templates.index(record["instruction"]))] += 1
    assert set(combos) == {("point", 0), ("point", 1), ("span", 0), ("span", 1)}


def test_deleted_text_never_starts_or_ends_with_punctuation() -> None:
    row = make_row("好的，我们走吧。明天见！", "zh")
    for candidate in candidates(row, "l_t_d"):
        start, end = candidate["edit"]["region"]
        assert CONTENT_CHAR_RE.match(row["text"][start]) and CONTENT_CHAR_RE.match(row["text"][end - 1])
