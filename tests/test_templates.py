import random

import pytest

from voxcanvas.annotate.templates import (
    ANCHORED_DELETION,
    D1,
    D4,
    DELETION,
    I1,
    I2,
    INSERTION,
    L_D,
    L_I,
    L_M,
    L_T_D,
    L_T_I,
    L_T_M,
    M1,
    M2,
    MULTI_BARE_DELETION,
    SINGLE_BARE_DELETION,
    SUBSTITUTION,
    deletion_reverse,
    format_time,
    insertion_reverse,
    ordinal,
)
from voxcanvas.text import apply_edits, content_key, edit_points


def test_every_template_uses_known_placeholders() -> None:
    pending = [DELETION, SUBSTITUTION, INSERTION, ANCHORED_DELETION, SINGLE_BARE_DELETION, MULTI_BARE_DELETION]
    pending += [D1, D4, M1, M2, I1, I2, L_D, L_M, L_I, L_T_D, L_T_M, L_T_I]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            rendered = item.format(x="X", y="Y", ord="O", t="1.0", ts="1.0", te="2.0")
            assert "{" not in rendered and "}" not in rendered, item
        else:
            pending += item.values() if isinstance(item, dict) else item


def test_ordinal() -> None:
    assert [ordinal(i, "zh") for i in (1, 2, 8)] == ["一", "二", "八"]
    assert [ordinal(i, "en") for i in (1, 3, 8)] == ["first", "third", "eighth"]
    with pytest.raises(AssertionError):
        ordinal(9, "en")


def test_format_time() -> None:
    assert format_time(3.16) == "3.2"
    assert format_time(4.72) == "4.7"
    assert format_time(0.04) == "0.0"


def test_insertion_reverse_partial_anchor() -> None:
    edited = "I think the new plan is great."
    assert insertion_reverse(edited, "plan", "new", "before", "en") == "Delete the 'new'."
    assert insertion_reverse(edited, "plan", "new", "before", "en", random.Random(0)) in (
        "Delete 'new'.",
        "Delete the 'new'.",
    )
    edited = "the new plan and the new idea"
    assert insertion_reverse(edited, "plan", "new", "before", "en") == "Delete the 'new' before 'plan'."
    assert insertion_reverse("new plan, new plan", "plan", "new", "before", "en") is None
    assert insertion_reverse("模仿完全新东方，完全不同", "模仿", "完全", "after", "zh") == "删除‘模仿’后面的‘完全’"
    assert insertion_reverse("我们完全模仿新东方", "模仿", "完全", "before", "zh") == "删除‘完全’"


def test_deletion_reverse_word_level_zh() -> None:
    raw = "最后一句歌词送给大家：把握生命里的每一分钟，全力以赴。我们心中的梦，没有人能随随便便成功。"
    edits = [{"op": "deletion", "new": "", "region": [raw.index("的梦"), raw.index("人能")]}]
    edited = apply_edits(raw, edits, "zh")
    (point,) = edit_points(raw, edited, edits)
    contexts = set()
    for seed in range(8):
        instruction, context, relation = deletion_reverse(edited, point, "的梦，没有", "zh", random.Random(seed), raw)
        assert instruction == INSERTION[relation]["zh"].format(x=context, y="的梦，没有")
        contexts.add(context)
    assert contexts == {"心中", "人能"}


def test_deletion_reverse_en_restores_raw() -> None:
    raw = "Wait, you belong there, right? That is what you said."
    edits = [{"op": "deletion", "new": "", "region": [6, 22]}]
    edited = apply_edits(raw, edits, "en")
    (point,) = edit_points(raw, edited, edits)
    instruction, context, relation = deletion_reverse(edited, point, "you belong there", "en", random.Random(0), raw)
    assert context in ("Wait", "right")
    start = edited.index(context)
    reverse = {
        "op": "insertion",
        "new": "you belong there",
        "direction": relation,
        "region": [start, start + len(context)],
    }
    assert content_key(apply_edits(edited, [reverse], "en")) == content_key(raw)


def test_deletion_reverse_grows_until_unique_and_checks_raw() -> None:
    edited = "we go and we go home"
    instruction, context, relation = deletion_reverse(edited, 9, "fast", "en", random.Random(3))
    assert edited.count(context) == 1
    assert deletion_reverse("a b a b", 3, "x", "en", random.Random(0)) is None
    assert deletion_reverse("we go home", 2, "really", "en", random.Random(0), raw_text="we go really home") is None
