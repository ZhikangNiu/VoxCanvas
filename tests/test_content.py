import random
import re
from itertools import pairwise, permutations

import pytest
from instruction_oracle import SPLIT, check_record, parse_instruction

from voxcanvas.annotate.content import (
    MAX_LENGTH_DELTA,
    complete_candidate,
    make_candidate,
    replay_reverse,
    select_span,
    unique_anchor,
)
from voxcanvas.annotate.llm import MAX_DRAWS
from voxcanvas.annotate.prompts import LANGUAGE_HINT
from voxcanvas.annotate.records import MIN_KEEP_TOKENS, build_record
from voxcanvas.annotate.templates import MULTI_BARE_DELETION
from voxcanvas.text import content_key, count_occurrences, segment_ids, tokenize_content


EN_SENTENCES = (
    "I think the plan is good, but the budget is not ready yet, so we wait until Monday.",
    "We went to the old market today and bought fresh bread, cheese and a bottle of wine.",
    "Honestly, I don't know why the train was late again this morning.",
    "The weather in the north is cold, and the weather in the south is warm all year.",
    "Well, you can help my mother and you can help me too. What happened to you?",
)
ZH_SENTENCES = (
    "最后一句歌词送给大家：把握生命里的每一分钟，全力以赴。我们心中的梦，没有人能随随便便成功。",
    "从新东方的基层做到中层，做到高层，做到总裁，然后出去创业了，跟谁学？嗯，完全模仿新东方。",
    "今天天气很好，我们一起去公园散步吧，顺便买点水果回家。",
    "这个问题其实很简单，只要你认真听课，按时完成作业，成绩自然就会提高。",
    "他说他明天要去北京开会，后天再回上海，所以周末没有时间。",
)
WORDS = {
    "en": ("zebra", "quietly", "violet", "marble", "orbit", "lantern", "copper", "meadow", "velvet", "harbor"),
    "zh": ("斑马", "安静地", "紫罗兰", "大理石", "轨道", "灯笼", "青铜", "草原", "丝绒", "港口"),
}


class WordBankLLM:
    """Fake LLM: replies[target] lists answers for prompts about that target; other prompts get fresh words."""

    def __init__(self, replies: dict[str, list[str]] | None = None) -> None:
        self.replies = replies or {}
        self.num_calls = 0

    def complete(self, prompt: str, system: str) -> str:
        self.num_calls += 1
        target = re.search(r"^target: (.*)$", prompt, re.MULTILINE).group(1)
        if self.replies.get(target):
            return self.replies[target].pop(0)
        words = WORDS["zh" if system == LANGUAGE_HINT["zh"] else "en"]
        return words[self.num_calls % len(words)]


def make_row(text: str, lang: str, uuid: str = "u0") -> dict:
    return {"uuid": uuid, "audio_path": f"{uuid}.mp3", "language": lang, "text": text, "align_word": [["x", 0.0, 0.1]]}


def edit_at(text: str, op: str, target: str, direction: str = "") -> dict:
    start = text.index(target)
    return {"op": op, "target": target, "new": "", "direction": direction, "region": [start, start + len(target)]}


def run(text: str, lang: str, subset: str, edits: list[dict], llm: WordBankLLM | None, seed: int = 0) -> dict | None:
    candidate = {"row": make_row(text, lang), "subset": subset, "length_bucket": "8-15", "edits": edits}
    candidate["num_edits_target"] = len(edits)
    return complete_candidate(candidate, llm, random.Random(seed))


def token_range(tokens: list[tuple[str, int, int]], start: int, end: int) -> tuple[int, int]:
    inside = [i for i, (_, s, e) in enumerate(tokens) if s >= start and e <= end]
    assert inside and tokens[inside[0]][1] == start and tokens[inside[-1]][2] == end
    return inside[0], inside[-1] + 1


def test_select_span_growth_uniqueness_and_cap() -> None:
    text = "I like it and I like you"
    tokens = tokenize_content(text, "en")
    assert select_span(text, tokens, 6, 1, "en") == (6, 7)
    assert select_span(text, tokens, 1, 1, "en") == (0, 3)  # like -> I like (left) -> I like it (right)
    assert select_span(text, tokens, 5, 1, "en") == (4, 7)  # like -> I like (left) -> I like you (right)
    for text, idx, expected in (("The cat and the dog", 3, (2, 4)), ("the weather is nice today", 0, (0, 1))):
        assert select_span(text, tokenize_content(text, "en"), idx, 1, "en") == expected  # any case, whole words
    text = "对对对对，我们走吧，回家去"
    tokens = [("对", i, i + 1) for i in range(4)] + [(w, s + 4, e + 4) for w, s, e in tokenize_content(text[4:], "zh")]
    # counted with overlaps, "对对对" occurs twice (non-overlapping str.count would call it unique)
    assert select_span(text, tokens, 1, 2, "zh") == (0, 4)
    text = "alpha beta gamma delta"
    assert select_span(text, tokenize_content(text, "en"), 0, 3, "en") == (0, 2)  # 3 tokens exceed 60 % of the chars
    assert select_span("ha ha", tokenize_content("ha ha", "en"), 0, 1, "en") is None


def test_single_deletion_keeps_min_keep_tokens() -> None:
    text = "Oh, I am exhausted."  # "Oh, I am" fits within 60 % of the chars, so only the token cap stops it
    tokens = tokenize_content(text, "en")
    kept = set()
    for seed in range(40):
        (edit,) = make_candidate(make_row(text, "en"), "deletion", random.Random(seed))["edits"]
        lo, hi = token_range(tokens, *edit["region"])
        kept.add(len(tokens) - (hi - lo))
    assert min(kept) == MIN_KEEP_TOKENS


def test_unique_anchor_grows_away_from_insertion_point_within_segment() -> None:
    for text, lang, cases in (
        ("go home now, go there", "en", [(3, "before", (3, 5)), (3, "after", None), (0, "before", (0, 2))]),
        ("我们去北京，我们去上海", "zh", [(4, "after", None), (4, "before", (4, 6)), (2, "after", (2, 3))]),
    ):
        tokens = tokenize_content(text, lang)
        for idx, direction, expected in cases:
            assert unique_anchor(text, tokens, segment_ids(text, tokens), idx, direction, lang) == expected


@pytest.mark.parametrize("subset", ["deletion", "substitution", "insertion"])
def test_single_candidates_are_unique_and_capped(subset: str) -> None:
    for lang, sentences in (("en", EN_SENTENCES), ("zh", ZH_SENTENCES)):
        for text in sentences:
            tokens = tokenize_content(text, lang)
            for seed in range(40):
                candidate = make_candidate(make_row(text, lang), subset, random.Random(seed))
                if candidate is None:
                    assert subset == "insertion"
                    continue
                (edit,) = candidate["edits"]
                assert candidate["length_bucket"] == ("8-15" if len(tokens) < 16 else "16+")
                assert count_occurrences(text, edit["target"], lang) == 1
                start, end = edit["region"]
                if subset != "insertion":
                    assert text[start:end] == edit["target"] and end - start <= 0.6 * len(text)
                    token_range(tokens, start, end)
                    continue
                assert (text[start:end], start, end) in tokens
                assert not re.search(r"[，,。.！!？?；;：:]", edit["target"])
                before = edit["direction"] == "before"
                assert text.startswith(edit["target"], start) if before else text[:end].endswith(edit["target"])


def test_multi_placement_keeps_two_token_gaps() -> None:
    for lang, sentences in (("en", EN_SENTENCES), ("zh", ZH_SENTENCES)):
        for text in sentences:
            tokens = tokenize_content(text, lang)
            for seed in range(40):
                candidate = make_candidate(make_row(text, lang), "multi", random.Random(seed))
                if candidate is None:
                    continue
                assert 2 <= len(candidate["edits"]) <= candidate["num_edits_target"] <= 6
                ranges = []
                for edit in candidate["edits"]:
                    start, end = edit["region"]
                    if edit["op"] == "insertion" and edit["direction"] == "before":
                        end = start + len(edit["target"])
                    elif edit["op"] == "insertion":
                        start = end - len(edit["target"])
                    ranges.append(token_range(tokens, start, end))
                assert ranges == sorted(ranges)
                assert all(right[0] - left[1] >= 2 for left, right in pairwise(ranges))


def test_deletion_reverse_uses_word_level_context() -> None:
    text = ZH_SENTENCES[0]
    edits = [edit_at(text, "deletion", "的梦，没有")]
    reverses = {run(text, "zh", "deletion", edits, None, seed)["reverse_instruction"] for seed in range(8)}
    assert reverses == {"在‘心中’后面加上‘的梦，没有’", "在‘人能’前面加上‘的梦，没有’"}
    text = "Wait, you belong there, right? I think so."
    record = run(text, "en", "deletion", [edit_at(text, "deletion", "you belong there")], None)
    assert record["edited_text"] == "Wait, right? I think so."
    assert record["reverse_instruction"] in (
        "Add 'you belong there' after 'Wait'.",
        "Add 'you belong there' before 'right'.",
    )
    assert check_record(record) == "ok"


def test_substitution_redraws_and_builds_record() -> None:
    text = "We went to the old market today and bought fresh bread."
    edits = [edit_at(text, "substitution", "old market")]
    llm = WordBankLLM({"old market": ["went went", "new mall"]})
    record = run(text, "en", "substitution", edits, llm)
    assert llm.num_calls == 2  # "went went" passes the word validators but adds an adjacent duplicate
    expected_edit = {
        "op": "substitution",
        "target": "old market",
        "new": "new mall",
        "direction": "",
        "region": [15, 25],
    }
    assert record == build_record(
        make_row(text, "en"),
        "single_span",
        "substitution",
        "substitution",
        "We went to the new mall today and bought fresh bread.",
        "Replace 'old market' with 'new mall'.",
        "Replace 'new mall' with 'old market'.",
        [expected_edit],
    )
    assert record["id"] == "single_span/substitution/u0" and record["meta"] == {}
    assert run(text, "en", "substitution", edits, WordBankLLM({"old market": ["went"] * MAX_DRAWS})) is None


def test_single_span_redraws_overlong_or_irreversible_answers() -> None:
    text = "We went to the old market today and bought fresh bread."
    words = [f"word{i}" for i in range(2 + MAX_LENGTH_DELTA + 1)]
    longest = " ".join(words[:-1])  # two target words plus MAX_LENGTH_DELTA
    llm = WordBankLLM({"old market": [" ".join(words), longest]})
    record = run(text, "en", "substitution", [edit_at(text, "substitution", "old market")], llm)
    assert (record["edits"][0]["new"], llm.num_calls) == (longest, 2)
    # swapping back hits the copy of "so much so" that starts in the original "so much", not the edit
    text = "I love you so much more than words."
    llm = WordBankLLM({"more": ["so much so", "deeply"]})
    record = run(text, "en", "substitution", [edit_at(text, "substitution", "more")], llm)
    assert (record["edited_text"], llm.num_calls) == ("I love you so much deeply than words.", 2)
    # the two copies of "说话说" overlap, so neither the bare nor the anchored reverse of "说" is unambiguous
    text = "他很会说话话说回来他也很累"
    llm = WordBankLLM({"说话": ["说", "得体"]})
    record = run(text, "zh", "insertion", [edit_at(text, "insertion", "说话", "after")], llm)
    assert (record["reverse_instruction"], llm.num_calls) == ("删除‘得体’", 2)
    assert check_record(record) == "ok"


def test_insertion_reverse_uses_partial_anchor() -> None:
    cases = [
        ("I think the plan is good.", "en", "plan", "new", "Add 'new' before 'plan'.", "Delete the 'new'."),
        # "really" already occurs, so the reverse deletion keeps its anchor
        ("I really think it is good.", "en", "good", "really", "Add 'really' before 'good'.",
         "Delete the 'really' before 'good'."),
        ("今天天气很好，我们一起去公园散步吧。", "zh", "公园", "附近的", "在‘公园’前面加上‘附近的’", "删除‘附近的’"),
    ]  # fmt: skip
    for text, lang, anchor, new, forward, reverse in cases:
        record = run(
            text, lang, "insertion", [edit_at(text, "insertion", anchor, "before")], WordBankLLM({anchor: [new]})
        )
        assert (record["instruction"], record["reverse_instruction"]) == (forward, reverse)
        assert check_record(record) == "ok"
    # a grown anchor ("go there") keeps the selected token ("go") as the region
    edit = {"op": "insertion", "target": "go there", "new": "", "direction": "before", "region": [13, 15]}
    record = run("go home now, go there", "en", "insertion", [edit], WordBankLLM({"go there": ["slowly"]}))
    assert record["edited_text"] == "go home now, slowly go there"
    assert check_record(record) == "ok"


def test_multi_record_joins_forward_in_order_and_shuffles_reverse() -> None:
    text = "So I think the plan is good. So let's go home now."
    edits = [
        edit_at(text, "insertion", "plan", "before"),
        edit_at(text, "substitution", "good"),
        edit_at(text, "deletion", "let's"),
    ]
    records = [run(text, "en", "multi", edits, WordBankLLM({"plan": ["new"], "good": ["great"]}), s) for s in range(12)]
    record = records[0]
    assert record["edited_text"] == "So I think the new plan is great. So go home now."
    assert record["instruction"] == "Add 'new' before 'plan'. Replace 'good' with 'great'. Delete 'let's'."
    assert (record["id"], record["edit_type"], record["num_edits"]) == ("multi_span/multi/u0", "multi", 3)
    assert record["meta"] == {"num_edits_target": 3}
    orders = {tuple(kind for kind, _ in parse_instruction(r["reverse_instruction"], "en")) for r in records}
    assert orders == set(permutations(("add", "delete", "replace")))
    assert all(check_record(r) == "ok" and check_record(r, simultaneous=True) == "ok" for r in records)


def test_multi_zh_joins_with_comma_and_strips_leading_punctuation() -> None:
    text = "好的，我们明天去北京开会，后天再回上海。"
    edits = [edit_at(text, "deletion", "好的"), edit_at(text, "substitution", "北京")]
    record = run(text, "zh", "multi", edits, WordBankLLM({"北京": ["广州"]}))
    assert record["edited_text"] == "我们明天去广州开会，后天再回上海。"
    assert record["instruction"] == "删掉‘好的’，把‘北京’改成‘广州’"
    assert "把‘广州’改成‘北京’" in record["reverse_instruction"].split("，")
    assert check_record(record) == "ok"


def test_multi_partial_anchor_bare_and_anchored() -> None:
    text = "We really went to the market and bought some bread for the kids."
    edits = [edit_at(text, "insertion", "market", "before"), edit_at(text, "insertion", "kids", "before")]
    reverses = set()
    for seed in range(10):
        record = run(text, "en", "multi", edits, WordBankLLM({"market": ["old"], "kids": ["hungry"]}), seed)
        reverses.update(re.split(SPLIT["en"], record["reverse_instruction"]))
        assert check_record(record) == "ok"
    assert reverses == {template.format(y=y) for template in MULTI_BARE_DELETION["en"] for y in ("old", "hungry")}
    record = run(text, "en", "multi", edits, WordBankLLM({"market": ["really"], "kids": ["hungry"]}))
    assert "Delete the 'really' before 'market'." in record["reverse_instruction"]


def test_reverse_chain_replay() -> None:
    raw = "So I think the plan is good. So let's go home now."
    edits = [
        {"op": "insertion", "target": "plan", "new": "new", "direction": "before", "region": [15, 19]},
        {"op": "substitution", "target": "good", "new": "great", "direction": "", "region": [23, 27]},
        {"op": "deletion", "target": "let's", "new": "", "direction": "", "region": [32, 37]},
    ]
    edited = "So I think the new plan is great. So go home now."
    assert content_key(replay_reverse(edited, edits, [None, None, ("go", "before")], "en")) == content_key(raw)
    assert replay_reverse(edited, edits, [None, None, ("missing", "after")], "en") is None
    # both substitutions become "Zed": swapping back globally cannot restore the sentence, so the sample is dropped
    text = "Alpha went home, beta stayed at work, and gamma left early."
    edits = [edit_at(text, "substitution", "Alpha"), edit_at(text, "substitution", "gamma")]
    assert run(text, "en", "multi", edits, WordBankLLM({"Alpha": ["Zed"], "gamma": ["Zed"]})) is None


def test_multi_drops_substitution_whose_reverse_is_ambiguous() -> None:
    text = "So I think the plan is good. So let's go home now."
    edits = [edit_at(text, "substitution", "plan"), edit_at(text, "insertion", "home", "before")]
    # the insertion repeats the substitution's new text, so "Replace 'violet' with 'plan'." hits both copies
    assert run(text, "en", "multi", edits, WordBankLLM({"plan": ["violet"], "home": ["violet"]})) is None
    record = run(text, "en", "multi", edits, WordBankLLM({"plan": ["violet"], "home": ["marble"]}))
    assert record["edited_text"] == "So I think the violet is good. So let's go marble home now."
    assert check_record(record) == "ok"


def test_multi_drops_new_adjacent_duplicate() -> None:
    text = "So I think the plan is good. So let's go home now."
    edits = [edit_at(text, "insertion", "plan", "before"), edit_at(text, "deletion", "let's")]
    # multi-span answers are not redrawn: "the new" before "plan" reads "the the new plan"
    assert run(text, "en", "multi", edits, WordBankLLM({"plan": ["the new"]})) is None
    record = run(text, "en", "multi", edits, WordBankLLM({"plan": ["new"]}))
    assert record["edited_text"] == "So I think the new plan is good. So go home now."
    assert check_record(record) == "ok"


def test_multi_drops_deletion_context_that_reaches_into_a_later_edit() -> None:
    text = "Hi there, we go home and we go out."
    edits = [edit_at(text, "deletion", "Hi there"), edit_at(text, "insertion", "home", "before")]
    # the head deletion has no left context and "we go" occurs twice, so its context grows to "we go zebra",
    # which the right-to-left replay removes before restoring the deletion
    for seed in range(4):
        assert run(text, "en", "multi", edits, WordBankLLM({"home": ["zebra"]}), seed) is None
    edits[1] = edit_at(text, "insertion", "out", "before")
    record = run(text, "en", "multi", edits, WordBankLLM({"out": ["zebra"]}))
    assert "Add 'Hi there' before 'we go home'." in record["reverse_instruction"]
    assert check_record(record) == "ok"


def test_oracle_rejects_broken_records() -> None:
    text = "So I think the plan is good. So let's go home now."
    edits = [edit_at(text, "substitution", "good"), edit_at(text, "deletion", "let's")]
    record = run(text, "en", "multi", edits, WordBankLLM({"good": ["great"]}))
    assert check_record(record) == "ok"
    assert check_record({**record, "edited_text": record["edited_text"].replace("home", "")}) == "forward_content"
    assert check_record({**record, "instruction": "Replace 'So' with 'great'. Delete 'let's'."}) == "forward_not_unique"
    broken = record["reverse_instruction"].replace("Replace 'great' with 'good'.", "Replace 'great' with 'fine'.")
    assert check_record({**record, "reverse_instruction": broken}) == "reverse_unmatched"
    context = re.search(r"Add 'let's' (?:before|after) '([^']+)'", record["reverse_instruction"]).group(1)
    broken = record["reverse_instruction"].replace(f"'{context}'", "'So'")
    assert check_record({**record, "reverse_instruction": broken}) == "add_ambiguous"


@pytest.mark.parametrize("subset", ["deletion", "substitution", "insertion", "multi"])
def test_generated_records_pass_the_oracle(subset: str) -> None:
    task = "multi_span" if subset == "multi" else "single_span"
    num_records = 0
    for lang, sentences in (("en", EN_SENTENCES), ("zh", ZH_SENTENCES)):
        for i, text in enumerate(sentences):
            for seed in range(15):
                candidate = make_candidate(make_row(text, lang, f"{lang}{i}"), subset, random.Random(seed))
                record = candidate and complete_candidate(candidate, WordBankLLM(), random.Random(seed))
                if record is None:
                    continue
                num_records += 1
                assert (record["id"], record["task"], record["subset"]) == (f"{task}/{subset}/{lang}{i}", task, subset)
                assert record["edits"] == sorted(record["edits"], key=lambda edit: edit["region"][0])
                assert check_record(record) == "ok" and check_record(record, simultaneous=True) == "ok", record
    assert num_records >= 100
