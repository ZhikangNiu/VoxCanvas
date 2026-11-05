import re

from voxcanvas.text import content_key, count_occurrences, find_occurrences


# Instruction grammar of the oracle: (kind, pattern); x = target / anchor / context, y = new / deleted text.
FORMS = {
    "en": (
        ("delete_next", r"Delete the '(?P<y>.+)' (?P<d>before|after) '(?P<x>.+)'\."),
        ("delete", r"Delete (?:the )?'(?P<y>.+)'\."),
        ("replace", r"Replace '(?P<x>.+)' with '(?P<y>.+)'\."),
        ("add", r"Add '(?P<y>.+)' (?P<d>before|after) '(?P<x>.+)'\."),
    ),
    "zh": (
        ("delete_next", r"删除‘(?P<x>.+)’(?P<d>前面|后面)的‘(?P<y>.+)’"),
        ("delete", r"(?:删掉|删除)‘(?P<y>.+)’"),
        ("replace", r"把‘(?P<x>.+)’改成‘(?P<y>.+)’"),
        ("add", r"在‘(?P<x>.+)’(?P<d>前面|后面)加上‘(?P<y>.+)’"),
    ),
}
SPLIT = {"en": r"(?<=\.) (?=(?:Add|Replace|Delete) (?:the )?')", "zh": r"(?<=’)，(?=(?:在|把|删除|删掉)‘)"}
DIRECTIONS = {"前面": "before", "后面": "after", "before": "before", "after": "after"}
FORWARD_KIND = {"deletion": "delete", "substitution": "replace", "insertion": "add"}


def parse_instruction(instruction: str, lang: str) -> list[tuple[str, dict]]:
    """(kind, fields) of every sub-instruction; kinds delete / delete_next / replace / add."""
    actions = []
    for part in re.split(SPLIT[lang], instruction):
        kind, found = next((kind, m) for kind, pattern in FORMS[lang] if (m := re.fullmatch(pattern, part)))
        fields = found.groupdict()
        if "d" in fields:
            fields["d"] = DIRECTIONS[fields["d"]]
        actions.append((kind, fields))
    return actions


def adjacent_pair(edit: dict, lang: str) -> str:
    """Inserted text next to its anchor, as it reads in the edited text."""
    sep = " " if lang == "en" else ""
    return edit["new"] + sep + edit["target"] if edit["direction"] == "before" else edit["target"] + sep + edit["new"]


def splice(text: str, changes: list[tuple[int, int, str]]) -> str | None:
    """Apply (start, end, new) changes located on `text` all at once; None if two of them overlap."""
    last = len(text) + 1
    for start, end, new in sorted(changes, reverse=True):
        if end > last:
            return None
        text, last = text[:start] + f" {new} " + text[end:], start
    return text


def check_record(record: dict, simultaneous: bool = False) -> str:
    """Oracle on the instruction strings: "ok" or the first failed check.

    Forward sub-instructions name unique targets and give edited_text; reverse ones pair with the edits, have unique
    anchors and, replayed right to left (or all at once with `simultaneous`), restore original_text.
    """
    lang, raw, edited, edits = record["language"], record["original_text"], record["edited_text"], record["edits"]
    forward = parse_instruction(record["instruction"], lang)
    if [kind for kind, _ in forward] != [FORWARD_KIND[edit["op"]] for edit in edits]:
        return "forward_kinds"
    changes = []
    for (_, fields), edit in zip(forward, edits, strict=True):
        located = find_occurrences(raw, fields.get("x", fields["y"]), lang)
        if len(located) != 1:
            return "forward_not_unique"
        start, end = located[0]
        if edit["op"] == "insertion":
            before = edit["direction"] == "before"
            start = end = start if before else end
            fields_ok = (fields["x"], fields["y"], fields["d"]) == (edit["target"], edit["new"], edit["direction"])
            if not fields_ok or start != edit["region"][0 if before else 1]:
                return "forward_fields"
        elif [start, end] != edit["region"]:
            return "forward_fields"
        changes.append((start, end, "" if edit["op"] == "deletion" else fields["y"]))
    restored = splice(raw, changes)
    if restored is None or content_key(restored) != content_key(edited):
        return "forward_content"

    actions = [None] * len(edits)
    for kind, fields in parse_instruction(record["reverse_instruction"], lang):
        matches = [
            i
            for i, edit in enumerate(edits)
            if actions[i] is None
            and (
                (kind == "add" and edit["op"] == "deletion" and fields["y"] == edit["target"])
                or (
                    kind == "replace"
                    and edit["op"] == "substitution"
                    and (fields["x"], fields["y"]) == (edit["new"], edit["target"])
                )
                or (
                    kind in ("delete", "delete_next")
                    and edit["op"] == "insertion"
                    and fields["y"] == edit["new"]
                    and (kind == "delete" or (fields["x"], fields["d"]) == (edit["target"], edit["direction"]))
                )
            )
        ]
        if not matches:
            return "reverse_unmatched"
        actions[matches[0]] = (kind, fields)
    if None in actions:
        return "reverse_missing"
    changes = []
    for (kind, fields), edit in zip(actions, edits, strict=True):
        phrase = {"add": fields.get("x"), "delete": edit["new"], "delete_next": adjacent_pair(edit, lang)}.get(kind)
        if phrase is not None and count_occurrences(edited, phrase, lang) != 1:
            return f"{kind}_ambiguous"
        if kind == "replace":
            changes += [(start, end, edit["target"]) for start, end in find_occurrences(edited, edit["new"], lang)]
        else:
            start, end = find_occurrences(edited, phrase, lang)[0]
            if kind == "add":
                start = end = start if fields["d"] == "before" else end
            replacement = {"add": fields["y"], "delete": "", "delete_next": edit["target"]}[kind]
            changes.append((start, end, replacement))
    if simultaneous:
        text = splice(edited, changes)
        return "ok" if text is not None and content_key(text) == content_key(raw) else "reverse_content"

    text, sep = edited, " " if lang == "en" else ""
    for (kind, fields), edit in reversed(list(zip(actions, edits, strict=True))):
        if kind == "replace":
            text = text.replace(edit["new"], edit["target"])
        elif kind == "add":
            located = find_occurrences(text, fields["x"], lang)
            if not located:
                return "reverse_context_missing"
            point = located[0][0] if fields["d"] == "before" else located[0][1]
            text = text[:point] + sep + fields["y"] + sep + text[point:]
        else:
            for start, end in reversed(find_occurrences(text, adjacent_pair(edit, lang), lang)):
                text = text[:start] + edit["target"] + text[end:]
        text = re.sub(" {2,}", " ", text)
    return "ok" if content_key(text) == content_key(raw) else "reverse_content"
