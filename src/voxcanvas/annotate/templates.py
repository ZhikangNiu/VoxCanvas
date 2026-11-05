import random

from voxcanvas.text import apply_edits, content_key, count_occurrences, tokenize_content


# Placeholders: {x} = edited target / anchor / context, {y} = new text, {ord} = ordinal, {t} {ts} {te} = seconds.
INSTRUCTION_SEP = {"en": " ", "zh": "，"}
DELETION = {"zh": "删掉‘{x}’", "en": "Delete '{x}'."}
SUBSTITUTION = {"zh": "把‘{x}’改成‘{y}’", "en": "Replace '{x}' with '{y}'."}
INSERTION = {
    "before": {"zh": "在‘{x}’前面加上‘{y}’", "en": "Add '{y}' before '{x}'."},
    "after": {"zh": "在‘{x}’后面加上‘{y}’", "en": "Add '{y}' after '{x}'."},
}
# Reverse of an insertion: delete the inserted {y}, anchored on {x} only when {y} alone is ambiguous.
ANCHORED_DELETION = {
    "before": {"zh": "删除‘{x}’前面的‘{y}’", "en": "Delete the '{y}' before '{x}'."},
    "after": {"zh": "删除‘{x}’后面的‘{y}’", "en": "Delete the '{y}' after '{x}'."},
}
SINGLE_BARE_DELETION = {"zh": "删除‘{y}’", "en": "Delete the '{y}'."}
MULTI_BARE_DELETION = {"zh": ("删除‘{y}’", "删掉‘{y}’"), "en": ("Delete '{y}'.", "Delete the '{y}'.")}

D1 = {
    "zh": ("删除所有的‘{x}’", "把所有的‘{x}’都删掉", "删掉句子里所有的‘{x}’", "去掉全部的‘{x}’", "把所有‘{x}’删除"),
    "en": (
        "Delete all instances of '{x}'.",
        "Remove every '{x}'.",
        "Delete all the '{x}'.",
        "Remove all occurrences of '{x}'.",
        "Delete every '{x}'.",
    ),
}
D4 = {
    "zh": (
        "删除第{ord}个‘{x}’",
        "删掉第{ord}个‘{x}’",
        "把第{ord}个‘{x}’删掉",
        "请删掉第{ord}个‘{x}’",
        "请把第{ord}个‘{x}’删掉",
    ),
    "en": (
        "Delete the {ord} '{x}'.",
        "Remove the {ord} '{x}'.",
        "Please delete the {ord} '{x}'.",
        "Delete the {ord} '{x}' from the sentence.",
        "Remove the {ord} '{x}' please.",
    ),
}
M1 = {
    "zh": (
        "把所有的‘{x}’改成‘{y}’",
        "将句子里所有的‘{x}’都改成‘{y}’",
        "把全部的‘{x}’替换成‘{y}’",
        "所有的‘{x}’都改成‘{y}’",
        "把每一个‘{x}’都换成‘{y}’",
    ),
    "en": (
        'Replace all instances of "{x}" with "{y}".',
        'Change every "{x}" to "{y}".',
        'Replace all the "{x}" with "{y}".',
        'Turn every "{x}" into "{y}".',
        'Change all occurrences of "{x}" to "{y}".',
    ),
}
M2 = {
    "zh": (
        "把第{ord}个‘{x}’改成‘{y}’",
        "将第{ord}个‘{x}’改成‘{y}’",
        "把第{ord}个‘{x}’换成‘{y}’",
        "请把第{ord}个‘{x}’改成‘{y}’",
        "把句子里第{ord}个‘{x}’改成‘{y}’",
    ),
    "en": (
        'Change the {ord} "{x}" to "{y}".',
        'Replace the {ord} "{x}" with "{y}".',
        'Turn the {ord} "{x}" into "{y}".',
        'Please change the {ord} "{x}" to "{y}".',
        'Modify the {ord} "{x}" to "{y}".',
    ),
}
I1 = {
    "before": {
        "zh": ("在所有的‘{x}’前面加上‘{y}’", "把每个‘{x}’前面都加上‘{y}’", "在所有‘{x}’之前加上‘{y}’"),
        "en": ('Add "{y}" before every "{x}".', 'Insert "{y}" before each "{x}".', 'Put "{y}" before all "{x}".'),
    },
    "after": {
        "zh": ("在所有的‘{x}’后面加上‘{y}’", "把每个‘{x}’后面都加上‘{y}’", "在所有‘{x}’之后加上‘{y}’"),
        "en": ('Add "{y}" after every "{x}".', 'Insert "{y}" after each "{x}".', 'Put "{y}" after all "{x}".'),
    },
}
I2 = {
    "before": {
        "zh": ("在第{ord}个‘{x}’前面加上‘{y}’", "把第{ord}个‘{x}’前面加上‘{y}’", "在句子里第{ord}个‘{x}’前面加上‘{y}’"),
        "en": ('Add "{y}" before the {ord} "{x}".', 'Insert "{y}" before the {ord} "{x}".'),
    },
    "after": {
        "zh": ("在第{ord}个‘{x}’后面加上‘{y}’", "把第{ord}个‘{x}’后面加上‘{y}’", "在句子里第{ord}个‘{x}’后面加上‘{y}’"),
        "en": ('Add "{y}" after the {ord} "{x}".', 'Insert "{y}" after the {ord} "{x}".'),
    },
}
L_D = {
    "head": {
        "zh": ("删除句首的‘{x}’", "把句首的‘{x}’删掉", "去掉句子开头的‘{x}’"),
        "en": (
            'Delete the sentence-initial "{x}".',
            'Remove the first word "{x}".',
            'Delete "{x}" at the start of the sentence.',
        ),
    },
    "tail": {
        "zh": ("删除句尾的‘{x}’", "把句尾的‘{x}’删掉", "去掉句子结尾的‘{x}’"),
        "en": (
            'Delete the sentence-final "{x}".',
            'Remove the last word "{x}".',
            'Delete "{x}" at the end of the sentence.',
        ),
    },
}
L_M = {
    "head": {
        "zh": ("把句首的‘{x}’改成‘{y}’", "将句首的‘{x}’替换成‘{y}’", "把句子开头的‘{x}’改成‘{y}’"),
        "en": ('Change the sentence-initial "{x}" to "{y}".', 'Replace the first word "{x}" with "{y}".'),
    },
    "tail": {
        "zh": ("把句尾的‘{x}’改成‘{y}’", "将句尾的‘{x}’替换成‘{y}’", "把句子结尾的‘{x}’改成‘{y}’"),
        "en": ('Change the sentence-final "{x}" to "{y}".', 'Replace the last word "{x}" with "{y}".'),
    },
}
L_I = {
    ("head", "before"): {
        "zh": ("在句首的‘{x}’前面加上‘{y}’", "在句子开头的‘{x}’之前加上‘{y}’"),
        "en": ('Add "{y}" before the sentence-initial "{x}".', 'Insert "{y}" before the first word "{x}".'),
    },
    ("head", "after"): {
        "zh": ("在句首的‘{x}’后面加上‘{y}’", "在句子开头的‘{x}’之后加上‘{y}’"),
        "en": ('Add "{y}" after the sentence-initial "{x}".', 'Insert "{y}" after the first word "{x}".'),
    },
    ("tail", "before"): {
        "zh": ("在句尾的‘{x}’前面加上‘{y}’", "在句子结尾的‘{x}’之前加上‘{y}’"),
        "en": ('Add "{y}" before the sentence-final "{x}".', 'Insert "{y}" before the last word "{x}".'),
    },
    ("tail", "after"): {
        "zh": ("在句尾的‘{x}’后面加上‘{y}’", "在句子结尾的‘{x}’之后加上‘{y}’"),
        "en": ('Add "{y}" after the sentence-final "{x}".', 'Insert "{y}" after the last word "{x}".'),
    },
}
# Keys: (mode, has_word); has_word=False templates give only the time.
L_T_D = {
    ("point", True): {
        "zh": ("删除第{t}秒的‘{x}’", "把第{t}秒的‘{x}’删掉", "去掉第{t}秒的‘{x}’"),
        "en": ('Delete "{x}" at {t}s.', 'Remove "{x}" at the {t}s mark.', 'Delete the word "{x}" at second {t}.'),
    },
    ("point", False): {
        "zh": ("删除第{t}秒的内容", "把第{t}秒的内容删掉", "去掉第{t}秒处的内容"),
        "en": (
            "Delete the content at {t}s.",
            "Remove what is said at the {t}s mark.",
            "Delete the audio at second {t}.",
        ),
    },
    ("span", True): {
        "zh": ("删除第{ts}到{te}秒的‘{x}’", "把第{ts}到{te}秒的‘{x}’删掉", "去掉第{ts}至{te}秒的‘{x}’"),
        "en": ('Delete "{x}" from {ts}s to {te}s.', 'Remove "{x}" between {ts}s and {te}s.'),
    },
    ("span", False): {
        "zh": ("删除第{ts}到{te}秒的内容", "把第{ts}到{te}秒的内容删掉", "去掉第{ts}至{te}秒的内容"),
        "en": ("Delete the content from {ts}s to {te}s.", "Remove what is said between {ts}s and {te}s."),
    },
}
L_T_M = {
    ("point", True): {
        "zh": ("把第{t}秒的‘{x}’改成‘{y}’", "将第{t}秒的‘{x}’替换成‘{y}’"),
        "en": ('Change "{x}" at {t}s to "{y}".', 'Replace "{x}" at the {t}s mark with "{y}".'),
    },
    ("point", False): {
        "zh": ("把第{t}秒的内容改成‘{y}’", "将第{t}秒处的内容替换成‘{y}’"),
        "en": ('Change the content at {t}s to "{y}".', 'Replace what is said at {t}s with "{y}".'),
    },
    ("span", True): {
        "zh": ("把第{ts}到{te}秒的‘{x}’改成‘{y}’", "将第{ts}至{te}秒的‘{x}’替换成‘{y}’"),
        "en": ('Change "{x}" from {ts}s to {te}s into "{y}".', 'Replace "{x}" between {ts}s and {te}s with "{y}".'),
    },
    ("span", False): {
        "zh": ("把第{ts}到{te}秒的内容改成‘{y}’", "将第{ts}至{te}秒的内容替换成‘{y}’"),
        "en": (
            'Change the content from {ts}s to {te}s into "{y}".',
            'Replace what is said between {ts}s and {te}s with "{y}".',
        ),
    },
}
# Keys: (direction, has_word); "after the {t}s mark" means after the word whose midpoint is t.
L_T_I = {
    ("before", True): {
        "zh": ("在第{t}秒的‘{x}’前面加上‘{y}’", "在第{t}秒的‘{x}’之前插入‘{y}’"),
        "en": ('Add "{y}" before "{x}" at {t}s.', 'Insert "{y}" before "{x}" at the {t}s mark.'),
    },
    ("after", True): {
        "zh": ("在第{t}秒的‘{x}’后面加上‘{y}’", "在第{t}秒的‘{x}’之后插入‘{y}’"),
        "en": ('Add "{y}" after "{x}" at {t}s.', 'Insert "{y}" after "{x}" at the {t}s mark.'),
    },
    ("before", False): {
        "zh": ("在第{t}秒前面加上‘{y}’", "在第{t}秒处之前插入‘{y}’"),
        "en": ('Add "{y}" before the {t}s mark.', 'Insert "{y}" just before {t}s.'),
    },
    ("after", False): {
        "zh": ("在第{t}秒后面加上‘{y}’", "在第{t}秒处之后插入‘{y}’"),
        "en": ('Add "{y}" after the {t}s mark.', 'Insert "{y}" just after {t}s.'),
    },
}

ZH_ORDINALS = ("一", "二", "三", "四", "五", "六", "七", "八")
EN_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")


def ordinal(i: int, lang: str) -> str:
    """1-based ordinal word: 一..八 (zh), first..eighth (en)."""
    assert 1 <= i <= len(EN_ORDINALS), i
    return (ZH_ORDINALS if lang == "zh" else EN_ORDINALS)[i - 1]


def format_time(seconds: float) -> str:
    return f"{seconds:.1f}"


def insertion_reverse(
    edited_text: str, anchor: str, new: str, direction: str, lang: str, rng: random.Random | None = None
) -> str | None:
    """Deletion instruction undoing the insertion of `new` next to `anchor` (partial anchor).

    The bare form is used when `new` occurs once in edited_text (single_span: fixed template; multi_span: drawn with
    `rng`), the anchored form when only `new` + `anchor` is unique, and None when even that is ambiguous.
    """
    if count_occurrences(edited_text, new, lang) == 1:
        bare = SINGLE_BARE_DELETION[lang] if rng is None else rng.choice(MULTI_BARE_DELETION[lang])
        return bare.format(y=new)
    sep = " " if lang == "en" else ""
    pair = new + sep + anchor if direction == "before" else anchor + sep + new
    if count_occurrences(edited_text, pair, lang) != 1:
        return None
    return ANCHORED_DELETION[direction][lang].format(x=anchor, y=new)


def deletion_reverse(
    edited_text: str, del_point: int, deleted: str, lang: str, rng: random.Random, raw_text: str | None = None
) -> tuple[str, str, str] | None:
    """Reverse of a deletion: (instruction, context, relation) adding `deleted` back next to a unique word context.

    `del_point` is an offset in edited_text (text.edit_points). With `raw_text`, the first unique context on a side
    must restore raw_text's content key, else that side is abandoned (context growth: docs/pipeline.md §4.3).
    """
    assert content_key(deleted), deleted
    relations = ["before", "after"]
    rng.shuffle(relations)
    for relation in relations:
        if relation == "before":
            words = [(del_point + s, del_point + e) for _, s, e in tokenize_content(edited_text[del_point:], lang)]
            spans = [(words[0][0], end) for _, end in words]
        else:
            words = [(s, e) for _, s, e in tokenize_content(edited_text[:del_point], lang)]
            spans = [(start, words[-1][1]) for start, _ in reversed(words)]
        for start, end in spans:
            context = edited_text[start:end]
            if count_occurrences(edited_text, context, lang) != 1:
                continue
            if raw_text is not None:
                reverse_edit = {"op": "insertion", "new": deleted, "direction": relation, "region": [start, end]}
                if content_key(apply_edits(edited_text, [reverse_edit], lang)) != content_key(raw_text):
                    break
            return INSERTION[relation][lang].format(x=context, y=deleted), context, relation
    return None
