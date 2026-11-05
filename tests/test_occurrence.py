import random

import pytest
from conftest import ScriptedLLM

from voxcanvas.annotate.llm import MAX_DRAWS
from voxcanvas.annotate.occurrence import (
    LENGTH_BOUNDS,
    MAX_NTH_COUNT,
    SUBSETS,
    complete_candidate,
    find_repeats,
    make_candidate,
)
from voxcanvas.annotate.records import length_bucket
from voxcanvas.annotate.templates import D1, D4, I1, I2, M1, M2
from voxcanvas.text import tokenize_aligned


def make_row(text: str, lang: str) -> dict:
    align_word = [[token, 0.0, 0.1] for token, _, _ in tokenize_aligned(text, lang)]
    return {"uuid": "u0", "audio_path": "u0.mp3", "language": lang, "text": text, "align_word": align_word}


def summary(text: str, lang: str) -> set[tuple[str, int, int]]:
    return {(unit["norm"], unit["count"], unit["num_tokens"]) for unit in find_repeats(text, lang)}


def annotate(
    text: str,
    lang: str,
    subset: str,
    *answers: str,
    ordinal: int | None = None,
    direction: str | None = None,
) -> tuple[dict | None, ScriptedLLM]:
    candidate = make_candidate(make_row(text, lang), subset, random.Random(0))
    assert candidate is not None
    if ordinal is not None:
        candidate["ordinal"] = ordinal
    if direction is not None:
        candidate["direction"] = direction
    llm = ScriptedLLM(list(answers))
    return complete_candidate(candidate, llm, random.Random(0)), llm


def test_find_repeats_keeps_maximal_units_only() -> None:
    units = summary("今天晚上吃了牛肉面，你今天中午吃了什么呢", "zh")
    assert ("今天", 2, 1) in units and ("吃了", 2, 2) in units
    assert ("吃", 2, 1) not in units and ("了", 2, 1) not in units
    assert summary("我喜欢你，我讨厌你", "zh") == {("我", 2, 1), ("你", 2, 1)}
    assert summary("I love you, I hate you", "en") == {("i", 2, 1), ("you", 2, 1)}


def test_find_repeats_counts_across_segments_without_crossing_punctuation() -> None:
    units = {unit["norm"]: unit for unit in find_repeats("今天晚上吃了牛肉面，你今天中午吃了什么呢", "zh")}
    assert units["吃了"]["occurrences"] == [[4, 6], [15, 17]]
    assert find_repeats("天气真好心情不错", "zh") == [] and find_repeats("abc def ghi", "en") == []
    assert all("," not in norm for norm, _, _ in summary("好,坏,好", "zh"))


def test_find_repeats_case_folding_and_surface() -> None:
    (unit,) = find_repeats("The cat sat. the dog ran.", "en")
    assert (unit["text"], unit["norm"], unit["occurrences"]) == ("The", "the", [[0, 3], [13, 16]])


def test_find_repeats_long_repeat_reported_once() -> None:
    (unit,) = find_repeats("a b c d e f g h , a b c d e f g h", "en")
    assert (unit["norm"], unit["count"], unit["num_tokens"]) == ("a b c d e f g h", 2, 8)


def test_find_repeats_shorter_unit_survives_with_other_neighbours() -> None:
    units = {unit["norm"]: unit["count"] for unit in find_repeats("今天晚上，今天早上，还有今天", "zh")}
    assert units["今天"] == 3
    units = {unit["norm"]: unit["count"] for unit in find_repeats("我爱牛肉面，他也爱牛肉面，但我更爱牛肉", "zh")}
    assert units["爱牛肉面"] == 2 and units["爱"] == 3 and units["我"] == 2
    assert "牛肉面" not in units and "牛肉" not in units


def test_find_repeats_spans_match_text() -> None:
    for text, lang in [
        ("今天晚上吃了牛肉面，你今天中午吃了什么呢", "zh"),
        ("I love you, I hate you", "en"),
        ("The cat sat. the dog ran.", "en"),
        ("我爱牛肉面，他也爱牛肉面，但我更爱牛肉", "zh"),
    ]:
        for unit in find_repeats(text, lang):
            for start, end in unit["occurrences"]:
                assert " ".join(text[start:end].split()).lower() == unit["norm"]


def test_find_repeats_keeps_embedded_english_words_whole() -> None:
    text = (
        "今天，It's a pleasure to be able to speak to you today. "
        "It's a pleasure to be able to speak to you today. 很高兴。"
    )
    longest = max(find_repeats(text, "zh"), key=lambda unit: unit["num_tokens"])
    assert longest["text"] == "It's a pleasure to be able to speak to you today"
    assert longest["norm"] == "it'sapleasuretobeabletospeaktoyoutoday"


def test_find_repeats_reports_overlapping_self_repeats() -> None:
    units = {unit["norm"]: unit for unit in find_repeats("啊，对对对对对对，土味情话，土味情话", "zh")}
    assert units["对对对对对"]["occurrences"] == [[2, 7], [3, 8]]


def test_make_candidate_picks_longest_disjoint_unit() -> None:
    row = make_row("啊，对对对对对对，土味情话，土味情话，我觉得那啊，这个算套路吗？", "zh")
    for subset in SUBSETS:
        candidate = make_candidate(row, subset, random.Random(0))
        assert candidate["unit"]["text"] == "土味情话" and candidate["length_bucket"] == "2"
        assert (candidate["ordinal"] is not None) == (subset in ("d4", "m2", "i2"))
        assert (candidate["direction"] in ("before", "after")) == subset.startswith("i")
    assert make_candidate(make_row("天气真好心情不错", "zh"), "d1", random.Random(0)) is None


def test_make_candidate_needs_every_visible_match_to_be_an_occurrence() -> None:
    hidden = make_row("婆婆说你浪费，你们隔阂大，你不再恭敬，爱挑你毛病。", "zh")
    assert all(make_candidate(hidden, subset, random.Random(0)) is None for subset in SUBSETS)
    assert make_candidate(make_row("It's fine, it works and it rocks", "en"), "d1", random.Random(0)) is None
    # 会 and 它 also hide inside the jieba word 它会, "we" inside "we're"
    for text, lang, unit in [
        ("但长期会低一点，它会如果它价格不变，它就会得到高于百分之七的无风险的收益率。", "zh", "的"),
        ("I'm pretty sure we we're not getting the best ending because we failed the dream sequence.", "en", "the"),
    ]:
        for subset in SUBSETS:
            assert make_candidate(make_row(text, lang), subset, random.Random(0))["unit"]["text"] == unit


def test_make_candidate_caps_ordinal_count() -> None:
    frequent = make_row(", ".join(["we can"] * (MAX_NTH_COUNT + 1)) + " today.", "en")
    assert make_candidate(frequent, "d1", random.Random(0))["unit"]["count"] == MAX_NTH_COUNT + 1
    assert make_candidate(frequent, "d4", random.Random(0)) is None


def test_make_candidate_draws_ordinal_in_range() -> None:
    row = make_row("我在北京工作，常去北京出差，北京很好", "zh")
    ordinals = {make_candidate(row, "d4", random.Random(seed))["ordinal"] for seed in range(50)}
    assert ordinals == {1, 2, 3}


def test_make_candidate_d1_needs_remaining_content() -> None:
    assert make_candidate(make_row("谢谢，谢谢，谢谢，谢谢。", "zh"), "d1", random.Random(0)) is None


def test_d1_deletes_every_occurrence() -> None:
    record, _ = annotate("翻炒均匀后，倒入之前炒好的土豆片，使调料均匀的裹在土豆上。", "zh", "d1")
    assert record["edited_text"] == "翻炒后，倒入之前炒好的土豆片，使调料的裹在土豆上。"
    assert record["id"] == "location_reference/d1/u0" and record["edit_type"] == "deletion"
    assert record["instruction"] in [template.format(x="均匀") for template in D1["zh"]]
    assert record["reverse_instruction"] is None and record["num_edits"] == 2
    assert [edit["region"] for edit in record["edits"]] == [[2, 4], [20, 22]]
    assert record["meta"] == {
        "locator": "all_occurrences",
        "target": "均匀",
        "count": 2,
        "occurrences": [[2, 4], [20, 22]],
    }


def test_d1_en_shows_surface_and_leaves_no_space_before_punctuation() -> None:
    text = (
        "Such as the paradox of the ship of Theseus, there is no paradox there. "
        "But we have in our mind this concept of the ship of Theseus, and as reality."
    )
    record, _ = annotate(text, "en", "d1")
    assert record["edited_text"] == (
        "Such as the paradox, there is no paradox there. But we have in our mind this concept, and as reality."
    )
    assert "'of the ship of Theseus'" in record["instruction"]


def test_d1_does_not_touch_lookalike_substrings() -> None:
    record, _ = annotate("the cat is there, the end", "en", "d1")
    assert record["edited_text"] == "cat is there, end"


def test_d4_deletes_the_ith_occurrence() -> None:
    text = "It was for a parent as frightening as it gets."
    record, _ = annotate(text, "en", "d4", ordinal=2)
    assert record["edited_text"] == "It was for a parent as frightening as gets."
    assert record["instruction"] in [template.format(ord="second", x="it") for template in D4["en"]]
    assert record["num_edits"] == 1 and record["edits"][0]["region"] == [38, 40]
    assert record["meta"]["locator"] == "nth_occurrence" and record["meta"]["ordinal"] == 2
    record, _ = annotate("有吗？有吗？快点没了我们就停了。", "zh", "d4", ordinal=2)
    assert record["edited_text"] == "有吗？快点没了我们就停了。"
    assert record["instruction"] in [template.format(ord="二", x="有吗") for template in D4["zh"]]


def test_d4_head_deletion_strips_orphan_punctuation() -> None:
    record, _ = annotate("有吗？有吗？快点没了我们就停了。", "zh", "d4", ordinal=1)
    assert record["edited_text"] == "有吗？快点没了我们就停了。"


def test_deletion_that_glues_aligned_tokens_is_dropped() -> None:
    text = "统计显示，从1985年到1990年间，日本企业总共21起500亿日元以上的海外并购案中，有18起是针对美国公司。"
    assert annotate(text, "zh", "d4", ordinal=1)[0] is None  # "21起500" -> "21500"
    assert annotate(text, "zh", "d1")[0] is None
    assert annotate(text, "zh", "d4", ordinal=2)[0]["edited_text"] == text.replace("18起", "18")


def test_m1_replaces_every_occurrence() -> None:
    record, llm = annotate("我在北京工作，常去北京出差", "zh", "m1", "工作", "上海")
    assert len(llm.prompts) == 2 and "target: 北京" in llm.prompts[0]
    assert record["edited_text"] == "我在上海工作，常去上海出差"
    assert record["instruction"] in [template.format(x="北京", y="上海") for template in M1["zh"]]
    assert record["edit_type"] == "substitution" and record["num_edits"] == 2
    assert [edit["new"] for edit in record["edits"]] == ["上海", "上海"]


def test_m1_rejects_case_variant_of_existing_word() -> None:
    record, _ = annotate("The cat and the dog", "en", "m1", "Cat", "one")
    assert record["edited_text"] == "one cat and one dog"
    assert record["instruction"] in [template.format(x="the", y="one") for template in M1["en"]]


def test_m1_drops_after_failed_redraws() -> None:
    record, llm = annotate("我在北京工作，常去北京出差", "zh", "m1", *(["北京"] * MAX_DRAWS))
    assert record is None and len(llm.prompts) == MAX_DRAWS


def test_m2_replaces_only_the_ith_occurrence() -> None:
    record, _ = annotate("the cat is there, the end", "en", "m2", "one", ordinal=2)
    assert record["edited_text"] == "the cat is there, one end"
    assert record["instruction"] in [template.format(ord="second", x="the", y="one") for template in M2["en"]]
    record, _ = annotate("我在北京工作，常去北京出差", "zh", "m2", "上海", ordinal=1)
    assert record["edited_text"] == "我在上海工作，常去北京出差"
    assert record["meta"]["ordinal"] == 1 and record["num_edits"] == 1


def test_target_is_the_surface_of_the_edited_occurrences() -> None:
    record, llm = annotate("The cat and the dog", "en", "m2", "one", ordinal=2)
    assert "target: the\n" in llm.prompts[0] and record["meta"]["target"] == "the"
    assert record["instruction"] in [template.format(ord="second", x="the", y="one") for template in M2["en"]]
    record, llm = annotate("The cat and the dog", "en", "m1", "one")
    assert "target: the\n" in llm.prompts[0] and record["meta"]["target"] == "the"
    record, _ = annotate("We met Theseus, and Theseus smiled", "en", "d1")
    assert "'Theseus'" in record["instruction"]


def test_i1_inserts_at_every_occurrence() -> None:
    record, llm = annotate("我在北京工作，常去北京出差", "zh", "i1", "美丽的", direction="before")
    assert "direction: before" in llm.prompts[0]
    assert record["edited_text"] == "我在美丽的北京工作，常去美丽的北京出差"
    assert record["instruction"] in [template.format(x="北京", y="美丽的") for template in I1["before"]["zh"]]
    assert [edit["direction"] for edit in record["edits"]] == ["before", "before"]
    record, _ = annotate("I fed the cat and you fed the cat", "en", "i1", "quickly", direction="after")
    assert record["edited_text"] == "I fed the cat quickly and you fed the cat quickly"


def test_i1_zh_inserts_next_to_mixed_case_latin_repeats() -> None:
    text = "我用ChatGPT写代码，也用chatgpt查资料，chatgpt很好用。"
    before, _ = annotate(text, "zh", "i1", "免费的", direction="before")
    after, _ = annotate(text, "zh", "i1", "免费的", direction="after")
    assert before["edited_text"] == "我用免费的ChatGPT写代码，也用免费的chatgpt查资料，免费的chatgpt很好用。"
    assert after["edited_text"] == "我用ChatGPT免费的写代码，也用chatgpt免费的查资料，chatgpt免费的很好用。"
    assert before["meta"]["target"] == "chatgpt" and before["num_edits"] == 3


def test_i1_en_rejects_text_already_next_to_a_target() -> None:
    text = "We click on the rental property calculator. Then click on rental property calculator."
    record, llm = annotate(text, "en", "i1", "the", "new", direction="before")
    assert len(llm.prompts) == 2
    assert record["edited_text"] == (
        "We click on the new rental property calculator. Then click on new rental property calculator."
    )


def test_i1_rejects_new_adjacent_duplicate() -> None:
    record, llm = annotate("I fed the cat and you fed the cat", "en", "i1", "you really", "gladly", direction="before")
    assert len(llm.prompts) == 2 and record["edits"][0]["new"] == "gladly"


def test_i2_inserts_at_the_ith_occurrence() -> None:
    record, _ = annotate("我在北京工作，常去北京出差", "zh", "i2", "美丽的", ordinal=2, direction="before")
    assert record["edited_text"] == "我在北京工作，常去美丽的北京出差"
    expected = [template.format(ord="二", x="北京", y="美丽的") for template in I2["before"]["zh"]]
    assert record["instruction"] in expected
    record, _ = annotate("the cat and the dog", "en", "i2", "big", ordinal=2, direction="before")
    assert record["edited_text"] == "the cat and big the dog"


@pytest.mark.parametrize(
    ("num_tokens", "label"), [(1, "1"), (5, "5"), (7, "6-7"), (9, "8-10"), (15, "11-15"), (68, "16+")]
)
def test_length_buckets(num_tokens: int, label: str) -> None:
    assert length_bucket(num_tokens, LENGTH_BOUNDS) == label
