import pytest

from voxcanvas.text import (
    apply_edits,
    content_key,
    count_occurrences,
    edit_keeps_alignment,
    edit_points,
    find_occurrences,
    gate,
    has_adjacent_duplicate,
    is_valid_insertion,
    is_valid_replacement,
    normalize_align_tokens,
    segment_ids,
    tokenize_aligned,
    tokenize_content,
    undo_insertion,
)


def deletion(text: str, phrase: str) -> dict:
    start = text.index(phrase)
    return {"op": "deletion", "target": phrase, "new": "", "direction": "", "region": [start, start + len(phrase)]}


def test_tokenize_content_keeps_latin_runs_whole_in_zh() -> None:
    text = "我觉得It's a good idea，真的2019年"
    tokens = tokenize_content(text, "zh")
    words = [word for word, _, _ in tokens]
    assert words[:6] == ["我", "觉得", "It's", "a", "good", "idea"]
    assert "2019" in words and "，" not in words
    assert all(text[start:end] == word for word, start, end in tokens)


def test_tokenize_content_en_keeps_apostrophes_and_hyphens() -> None:
    words = [word for word, _, _ in tokenize_content("It's a well-known fact, isn’t it?", "en")]
    assert words == ["It's", "a", "well-known", "fact", "isn’t", "it"]


def test_segment_ids_break_on_punctuation() -> None:
    text = "we go home, then we sleep. ok"
    assert segment_ids(text, tokenize_content(text, "en")) == [0, 0, 0, 1, 1, 1, 2]


def test_tokenize_aligned_matches_aligner_granularity() -> None:
    # The aligner keeps only ASCII ', so a curly apostrophe is dropped like any other mark.
    assert tokenize_aligned("It’s 5:00 a.m., a peer-reviewed ‘study’!", "en") == [
        ("its", 0, 4),
        ("500", 5, 9),
        ("am", 10, 13),
        ("a", 16, 17),
        ("peerreviewed", 18, 31),
        ("study", 33, 38),
    ]
    # zh splits Latin letters; digit runs stay whole as in the aligner; symbols and quotes are dropped.
    tokens = tokenize_aligned("我说，AI好+%在1,000米，‘两难’", "zh")
    assert [token for token, _, _ in tokens] == ["我", "说", "a", "i", "好", "在", "1000", "米", "两", "难"]
    assert tokens[6] == ("1000", 9, 14)


def test_normalize_align_tokens() -> None:
    assert normalize_align_tokens([["Don't", 0.0, 0.2], [" OK ", 0.2, 0.4]]) == ["don't", "ok"]


def make_row(text: str, tokens: list[str], language: str) -> dict:
    align_word = [[token, i * 0.2, i * 0.2 + 0.2] for i, token in enumerate(tokens)]
    return {"uuid": "u", "audio_path": "a.mp3", "language": language, "text": text, "align_word": align_word}


def test_gate() -> None:
    assert gate(make_row("Hello there, my old friend.", ["Hello", "there", "my", "old", "friend"], "en"))
    assert not gate(make_row("Hello there, my old friend.", ["Hello", "there", "my", "friend"], "en"))
    assert not gate(make_row("Hi there.", ["Hi", "there"], "en"))
    assert gate(make_row("twenty-five apples are here now", ["twentyfive", "apples", "are", "here", "now"], "en"))
    assert gate(make_row("It’s a nice day", ["Its", "a", "nice", "day"], "en"))
    assert not gate(make_row("It’s a nice day", ["It's", "a", "nice", "day"], "en"))
    # "us" holds the two content tokens U and S.
    assert not gate(make_row("We met in the U.S. last year", ["we", "met", "in", "the", "us", "last", "year"], "en"))
    assert gate(make_row("我们今天去北京。", list("我们今天去北京"), "zh"))
    assert gate(make_row("他在1991年‘出生’了", ["他", "在", "1991", "年", "出", "生", "了"], "zh"))
    assert not gate(make_row("我用AI写代码吧", ["我", "用", "AI", "写", "代", "码", "吧"], "zh"))
    assert not gate(make_row("I saw [UNK] there today.", ["I", "saw", "UNK", "there", "today"], "en"))
    # Utterances of another language that the corpus labelled en/zh; they align and have 4 CJK content tokens.
    japanese = "たとえ死のうと殺されよう、刑の知ったところではない。"
    japanese_tokens = ["たとえ", "死", "のうと", "殺", "されよう", "刑", "の", "知", "ったところではない"]
    assert not gate(make_row(japanese, japanese_tokens, "en"))
    assert not gate(make_row(japanese, japanese_tokens, "zh"))
    assert not gate(make_row("the result of a拆手", ["the", "result", "of", "a", "拆", "手"], "en"))
    with pytest.raises(ValueError):
        gate(make_row("Hola amigos", ["hola", "amigos"], "es"))


def test_content_key_keeps_uppercase_latin_in_zh() -> None:
    assert content_key("用ChatGPT来，写！") == "用chatgpt来写"
    assert content_key("It's OK.") == "itsok"


def test_apply_edits_deletion_en_swallows_space_and_collapses_punctuation() -> None:
    text = "Wait, you belong there, right? That is it."
    assert apply_edits(text, [deletion(text, "you belong there")], "en") == "Wait, right? That is it."
    text = "So I think it works."
    assert apply_edits(text, [deletion(text, "So")], "en") == "I think it works."


def test_apply_edits_deletion_zh_head_strips_leading_punctuation() -> None:
    text = "好的，我们走吧。"
    assert apply_edits(text, [deletion(text, "好的")], "zh") == "我们走吧。"
    text = "我们，好的。走吧"
    assert apply_edits(text, [deletion(text, "好的")], "zh") == "我们。走吧"


def test_apply_edits_substitution_and_insertion() -> None:
    text = "I like the red car."
    replace = {"op": "substitution", "target": "red", "new": "blue", "direction": "", "region": [11, 14]}
    assert apply_edits(text, [replace], "en") == "I like the blue car."
    before = {"op": "insertion", "target": "car", "new": "fast", "direction": "before", "region": [15, 18]}
    after = {"op": "insertion", "target": "car", "new": "today", "direction": "after", "region": [15, 18]}
    assert apply_edits(text, [before], "en") == "I like the red fast car."
    assert apply_edits(text, [after], "en") == "I like the red car today."
    zh = "我爱北京。"
    assert apply_edits(zh, [{"op": "insertion", "new": "首都", "direction": "before", "region": [2, 4]}], "zh") == (
        "我爱首都北京。"
    )


def test_apply_edits_multiple_mixed_edits() -> None:
    text = "So I think the plan is good. So let's go."
    edits = [
        {"op": "insertion", "target": "plan", "new": "new", "direction": "before", "region": [15, 19]},
        {"op": "substitution", "target": "good", "new": "great", "direction": "", "region": [23, 27]},
        {"op": "deletion", "target": "let's", "new": "", "direction": "", "region": [32, 37]},
    ]
    edited = apply_edits(text, edits, "en")
    assert edited == "So I think the new plan is great. So go."
    points = edit_points(text, edited, edits)
    assert [edited[:point].split()[-1] for point in points] == ["the", "is", "So"]
    with pytest.raises(AssertionError):
        apply_edits(text, [edits[1], {**edits[2], "region": [24, 30]}], "en")


def test_apply_edits_deletes_all_occurrences() -> None:
    text = "the cat and the dog, the end"
    edits = [{"op": "deletion", "new": "", "region": [m, m + 3]} for m in (0, 12, 21)]
    assert apply_edits(text, edits, "en") == "cat and dog, end"


def test_edit_points_after_punctuation_collapse() -> None:
    text = "我们心中的梦，没有人能成功。"
    edits = [deletion(text, "的梦，没有")]
    edited = apply_edits(text, edits, "zh")
    assert edited == "我们心中人能成功。"
    assert edit_points(text, edited, edits) == [4]


def test_edit_keeps_alignment() -> None:
    text = "I like the red car."
    for edit in (
        deletion(text, "red"),
        {"op": "substitution", "target": "red", "new": "blue", "direction": "", "region": [11, 14]},
        {"op": "insertion", "target": "car", "new": "fast", "direction": "before", "region": [15, 18]},
        {"op": "insertion", "target": "car", "new": "today", "direction": "after", "region": [15, 18]},
    ):
        assert edit_keeps_alignment(text, apply_edits(text, [edit], "en"), [edit], "en")
    # Deleting the 起 between two numbers glues them into one aligned token, "21500".
    zh = "日本企业总共21起500亿日元以上的并购案中，有18起是美国公司。"
    edits = [deletion(zh, "起")]
    edited = apply_edits(zh, edits, "zh")
    assert edited.startswith("日本企业总共21500亿")
    assert not edit_keeps_alignment(zh, edited, edits, "zh")
    # zh insertions add no space, so a digit answer merges with the number it follows.
    glue = {"op": "insertion", "target": "1991", "new": "500", "direction": "after", "region": [2, 6]}
    assert not edit_keeps_alignment("我在1991年出生。", apply_edits("我在1991年出生。", [glue], "zh"), [glue], "zh")
    # The aligner keeps a trailing ' with its word, so deleting "he" moves the mark onto "thinking".
    en = "That jump scare, thinking he' all funny."
    edits = [deletion(en, "he")]
    edited = apply_edits(en, edits, "en")
    assert edited == "That jump scare, thinking' all funny."
    assert not edit_keeps_alignment(en, edited, edits, "en")


def test_occurrences_are_case_insensitive_whole_word_and_overlapping() -> None:
    assert count_occurrences("The weather, the cat", "the", "en") == 2
    assert count_occurrences("very very very good", "very very", "en") == 2
    assert count_occurrences("对对对", "对对", "zh") == 2
    assert count_occurrences("我用AI，ai很好", "Ai", "zh") == 2
    assert find_occurrences("Go, so that, so", ", so", "en") == [(2, 6), (11, 15)]


def test_has_adjacent_duplicate() -> None:
    assert has_adjacent_duplicate("click on the the button", "en")
    assert has_adjacent_duplicate("The the end", "en")
    assert not has_adjacent_duplicate("then the end", "en")
    assert has_adjacent_duplicate("我们居然居然赢了", "zh")
    assert not has_adjacent_duplicate("我们赢了", "zh")


def test_is_valid_replacement() -> None:
    raw = "他说苹果很好吃"
    assert is_valid_replacement("香蕉", "苹果", raw, "zh")
    assert not is_valid_replacement("她", "他", raw, "zh")
    assert not is_valid_replacement("很好", "苹果", raw, "zh")
    assert not is_valid_replacement("香蕉。", "苹果", raw, "zh")
    assert not is_valid_replacement("，", "苹果", raw, "zh")
    assert not is_valid_replacement("banana", "苹果", raw, "zh")
    assert is_valid_replacement("GPU", "AI", "我用AI写", "zh")
    assert not is_valid_replacement("Apple", "apple", "an apple a day", "en")
    assert not is_valid_replacement("香蕉", "apple", "an apple a day", "en")
    assert is_valid_replacement("她们的", "他们", "他们来了", "zh")
    raw = "I bought an apple yesterday."
    assert is_valid_replacement("banana", "apple", raw, "en")
    for decorated in ("**banana**", "- banana", "# banana", "`banana`", "~banana~", "banana\u200b", "50%"):
        assert not is_valid_replacement(decorated, "apple", raw, "en")


def test_is_valid_insertion_checks_spaced_adjacency_case_insensitively() -> None:
    raw = "We went to The United States."
    assert not is_valid_insertion("the", "United", raw, "before", "en")
    assert is_valid_insertion("beautiful", "United", raw, "before", "en")
    assert not is_valid_insertion("美丽的", "United", raw, "before", "en")
    assert not is_valid_insertion("United", "United", raw, "before", "en")
    assert not is_valid_insertion("States now", "States", raw, "after", "en")
    assert not is_valid_insertion("首都", "北京", "我爱首都北京", "before", "zh")
    assert is_valid_insertion("美丽的", "北京", "我爱首都北京", "before", "zh")
    assert not is_valid_insertion("**beautiful**", "United", raw, "before", "en")


def test_undo_insertion() -> None:
    assert undo_insertion("I really like it and really like you", "like", "really", "before", "en") == (
        "I like it and like you"
    )
    assert undo_insertion("bathe like it", "like", "the", "before", "en") == "bathe like it"
    assert undo_insertion("我爱首都北京", "北京", "首都", "before", "zh") == "我爱北京"
    assert undo_insertion("我爱北京首都", "北京", "首都", "after", "zh") == "我爱北京"
    assert undo_insertion("I really Like it", "like", "really", "before", "en") == "I Like it"
    assert undo_insertion("我用免费的ChatGPT写代码，也用免费的chatgpt查资料", "chatgpt", "免费的", "before", "zh") == (
        "我用ChatGPT写代码，也用chatgpt查资料"
    )
