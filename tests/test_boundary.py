import random
from collections import Counter

import pytest
from conftest import ScriptedLLM, candidate_with, make_row

from voxcanvas.annotate import boundary
from voxcanvas.annotate.llm import MAX_DRAWS
from voxcanvas.annotate.prompts import GLOBAL_SUBSTITUTION_PROMPT, LOCATION_INSERTION_PROMPT
from voxcanvas.annotate.templates import L_D, L_I, L_M
from voxcanvas.utils import rng_for


def target_of(candidate: dict) -> str:
    start, end = candidate["edit"]["region"]
    assert candidate["edit"]["target"] == candidate["row"]["text"][start:end]
    return candidate["edit"]["target"]


@pytest.mark.parametrize(
    ("text", "lang", "head", "tail"),
    [
        ("基本上他们每天早上都要出去玩耍。", "zh", "基本上", "玩耍"),
        ("The money raised for the day was huge.", "en", "The", "huge"),
    ],
)
def test_edge_token_is_first_or_last_content_token(text: str, lang: str, head: str, tail: str) -> None:
    row = make_row(text, lang)
    for subset in ("l_m", "l_i"):
        assert target_of(candidate_with(boundary.make_candidate, row, subset, position="head")) == head
        assert target_of(candidate_with(boundary.make_candidate, row, subset, position="tail")) == tail
    assert candidate_with(boundary.make_candidate, row, "l_m", position="head")["length_bucket"] == "8-15"


def test_l_d_run_stays_in_edge_segment_and_keeps_two_tokens() -> None:
    row = make_row("Well, we went to the old market today.", "en")
    heads, tails = set(), Counter()
    for seed in range(300):
        candidate = boundary.make_candidate(row, "l_d", random.Random(seed))
        num_tokens = candidate["num_tokens"]
        expected = "6+" if num_tokens == boundary.MAX_DELETE_TOKENS else str(num_tokens)
        assert candidate["length_bucket"] == expected
        if candidate["position"] == "head":
            heads.add(target_of(candidate))
        else:
            tails[num_tokens] += 1
            assert "," not in target_of(candidate)
    assert heads == {"Well"}
    assert set(tails) == {1, 2, 3, 4, 5, 6}
    short = make_row("我们 一起 去，公园", "zh")
    for seed in range(100):
        candidate = boundary.make_candidate(short, "l_d", random.Random(seed))
        assert candidate["num_tokens"] <= 2


def test_l_d_head_zh_strips_orphan_punctuation() -> None:
    row = make_row("基本上，他们每天都出去。", "zh")
    record = boundary.complete_candidate(
        candidate_with(boundary.make_candidate, row, "l_d", position="head"), None, random.Random(0)
    )
    assert record["edited_text"] == "他们每天都出去。"
    assert record["instruction"] in [t.format(x="基本上") for t in L_D["head"]["zh"]]


def test_l_d_tail_en_keeps_final_period_and_head_has_no_leading_space() -> None:
    row = make_row("We must do everything we can to save lives.", "en")
    record = boundary.complete_candidate(
        candidate_with(boundary.make_candidate, row, "l_d", position="tail", num_tokens=1), None, random.Random(0)
    )
    assert record["edited_text"] == "We must do everything we can to save."
    row = make_row("The cat sat on the mat.", "en")
    record = boundary.complete_candidate(
        candidate_with(boundary.make_candidate, row, "l_d", position="head", num_tokens=1), None, random.Random(0)
    )
    assert record["edited_text"] == "cat sat on the mat."


def test_l_d_multiword_record() -> None:
    row = make_row("The quick brown fox jumps over the lazy dog.", "en")
    candidate = candidate_with(boundary.make_candidate, row, "l_d", position="head", num_tokens=3)
    record = boundary.complete_candidate(candidate, None, random.Random(0))
    assert record["id"] == "location_reference/l_d/u1"
    assert (record["task"], record["subset"], record["edit_type"]) == ("location_reference", "l_d", "deletion")
    assert record["edited_text"] == "fox jumps over the lazy dog."
    assert record["instruction"] in [t.format(x="The quick brown") for t in L_D["head"]["en"]]
    assert record["reverse_instruction"] is None
    assert record["edits"] == [
        {"op": "deletion", "target": "The quick brown", "new": "", "direction": "", "region": [0, 15]}
    ]
    assert record["num_edits"] == 1
    assert record["meta"] == {"locator": "sentence_position", "position": "head", "num_tokens": 3}
    assert candidate["length_bucket"] == "3"


def test_l_m_redraws_until_valid_and_uses_global_substitution_prompt() -> None:
    row = make_row("The cat sat on the mat.", "en")
    llm = ScriptedLLM(["mat", "One"])
    record = boundary.complete_candidate(
        candidate_with(boundary.make_candidate, row, "l_m", position="head"), llm, random.Random(0)
    )
    assert record["edited_text"] == "One cat sat on the mat."
    assert record["edit_type"] == "substitution"
    assert record["edits"][0] == {
        "op": "substitution",
        "target": "The",
        "new": "One",
        "direction": "",
        "region": [0, 3],
    }
    assert record["instruction"] in [t.format(x="The", y="One") for t in L_M["head"]["en"]]
    assert len(llm.prompts) == 2
    assert llm.prompts[0] == GLOBAL_SUBSTITUTION_PROMPT.format(raw_text=row["text"], target="The")


def test_l_m_rejects_new_adjacent_duplicate_and_drops_after_max_draws() -> None:
    row = make_row("I really like this old house", "en")
    candidate = candidate_with(boundary.make_candidate, row, "l_m", position="tail")
    record = boundary.complete_candidate(candidate, ScriptedLLM(["old car", "boat"]), random.Random(0))
    assert record["edited_text"] == "I really like this old boat"
    assert boundary.complete_candidate(candidate, ScriptedLLM(["old car"] * MAX_DRAWS), random.Random(0)) is None


@pytest.mark.parametrize(
    ("text", "lang", "position", "direction", "new", "edited"),
    [
        ("我们一起去公园玩耍", "zh", "head", "before", "其实", "其实我们一起去公园玩耍"),
        ("我们一起去公园玩耍", "zh", "tail", "after", "一下", "我们一起去公园玩耍一下"),
        ("The cat sat on the mat.", "en", "head", "after", "really", "The really cat sat on the mat."),
        ("The cat sat on the mat.", "en", "tail", "before", "old", "The cat sat on the old mat."),
    ],
)
def test_l_i_inserts_next_to_edge_token(
    text: str, lang: str, position: str, direction: str, new: str, edited: str
) -> None:
    row = make_row(text, lang)
    candidate = candidate_with(boundary.make_candidate, row, "l_i", position=position, direction=direction)
    llm = ScriptedLLM([new])
    record = boundary.complete_candidate(candidate, llm, random.Random(0))
    anchor = target_of(candidate)
    assert record["edited_text"] == edited
    assert record["edits"][0]["direction"] == direction and record["edits"][0]["target"] == anchor
    assert record["instruction"] in [t.format(x=anchor, y=new) for t in L_I[(position, direction)][lang]]
    assert llm.prompts[0] == LOCATION_INSERTION_PROMPT.format(
        raw_text=text, target=anchor, direction=direction, direction_zh={"before": "前面", "after": "后面"}[direction]
    )


def test_l_i_en_rejects_word_already_next_to_anchor() -> None:
    row = make_row("Let's call her tonight, okay?", "en")
    candidate = candidate_with(boundary.make_candidate, row, "l_i", position="head", direction="after")
    record = boundary.complete_candidate(candidate, ScriptedLLM(["call", "quickly"]), random.Random(0))
    assert record["edited_text"] == "Let's quickly call her tonight, okay?"


def test_template_choice_is_independent_of_position() -> None:
    row = make_row("The cat sat on the mat.", "en")
    combos = Counter()
    for i in range(200):
        uuid = f"u{i}"
        candidate = boundary.make_candidate(row, "l_m", rng_for(0, uuid, "l_m", "candidate"))
        record = boundary.complete_candidate(candidate, ScriptedLLM(["One"]), rng_for(0, uuid, "l_m", "complete"))
        templates = [t.format(x=target_of(candidate), y="One") for t in L_M[candidate["position"]]["en"]]
        combos[(candidate["position"], templates.index(record["instruction"]))] += 1
    assert set(combos) == {("head", 0), ("head", 1), ("tail", 0), ("tail", 1)}
