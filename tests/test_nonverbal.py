import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from conftest import write_jsonl
from f5_tts.model.utils import convert_char_to_pinyin
from qwen_asr.inference.qwen3_forced_aligner import ForcedAlignItem, ForcedAlignResult
from test_synthesize import SAMPLING, FakeCFM, FakeVocos
from test_verify import FakeASRModel

from voxcanvas import align
from voxcanvas.annotate import __main__ as cli
from voxcanvas.annotate.nonverbal import make_candidate, split_by_tag
from voxcanvas.synthesize import locate_nv_segments, synthesize_record
from voxcanvas.text import gate, strip_nv_tags, tokenize_aligned
from voxcanvas.utils import read_jsonl
from voxcanvas.verify import asr


def nv_row(transcript: str, lang: str, uuid: str = "u1") -> dict:
    """Aligned row of a tagged transcript; every aligned token lasts 0.1 s followed by a 0.3 s pause."""
    text = strip_nv_tags(transcript, lang)[0]
    tokens = [token for token, _, _ in tokenize_aligned(text, lang)]
    align_word = [[token, i * 0.4, i * 0.4 + 0.1] for i, token in enumerate(tokens)]
    return {
        "uuid": uuid,
        "audio_path": "a.wav",
        "language": lang,
        "transcript": transcript,
        "text": text,
        "align_word": align_word,
    }


def nv_record(row: dict, tag: str) -> dict:
    split = next(split for split in split_by_tag(row) if split["nv_tag"] == tag)
    candidate = make_candidate(split, "insertion", None)
    assert candidate is not None
    return candidate["record"]


@pytest.mark.parametrize(
    ("transcript", "lang", "text", "removed"),
    [
        ("Did we get[laugh]any pictures?", "en", "Did we get any pictures?", [("laugh", 10)]),
        ("then acting.[breath]I guess", "en", "then acting. I guess", [("breath", 12)]),
        ("[sigh] They say so [laugh].", "en", "They say so.", [("sigh", 0), ("laugh", 11)]),
        ("Hello [cough] , world", "en", "Hello, world", [("cough", 5)]),
        ("合影[breath]，[breath]也没有", "zh", "合影，也没有", [("breath", 2), ("breath", 3)]),
        ("[laugh] 你好[UNK]", "zh", "你好[UNK]", [("laugh", 0)]),
    ],
)
def test_strip_nv_tags(transcript: str, lang: str, text: str, removed: list[tuple[str, int]]) -> None:
    assert strip_nv_tags(transcript, lang) == (text, removed)


def test_strip_nv_tags_of_one_type_keeps_the_others() -> None:
    transcript = "Acting.[breath]I guess you might ask[laugh] me that.[laugh]"
    assert strip_nv_tags(transcript, "en", {"laugh"}) == (
        "Acting.[breath]I guess you might ask me that.",
        [("laugh", 36), ("laugh", 45)],
    )


def test_split_by_tag_suffixes_uuids_only_with_several_tag_types() -> None:
    row = {"uuid": "c1", "transcript": "a[laugh]b[breath]c[laugh]"}
    assert [(split["uuid"], split["nv_tag"]) for split in split_by_tag(row)] == [
        ("c1_laugh", "laugh"),
        ("c1_breath", "breath"),
    ]
    assert [split["uuid"] for split in split_by_tag({"uuid": "c1", "transcript": "a[laugh]b[laugh]"})] == ["c1"]
    assert split_by_tag({"uuid": "c1", "transcript": "no tags [UNK]"}) == []


# Released VoxCanvas-41M nonverbal rows: (original_text, nv_tag, language, synthetic_text, forward, reverse).
RELEASE_ROWS = [
    (
        "Martial arts schools and gyms and then acting.[breath]I guess you might ask me that in acting.[laugh]",
        "laugh",
        "en",
        "Martial arts schools and gyms and then acting.[breath]I guess you might ask me that in acting.",
        "Delete the '[laugh]' after 'that in acting'.",
        "Add '[laugh]' after 'that in acting'.",
    ),
    (
        "[laugh]Yeah.[breath]Oh, man, it was certain to give out today or at some point.[laugh]",
        "laugh",
        "en",
        "Yeah.[breath]Oh, man, it was certain to give out today or at some point.",
        "Delete the '[laugh]' before 'Yeah', delete the '[laugh]' after 'at some point'.",
        "Add '[laugh]' before 'Yeah', add '[laugh]' after 'at some point'.",
    ),
    (
        "I live in a society where we are. I couldn't describe it better.[breath][breath]",
        "breath",
        "en",
        "I live in a society where we are. I couldn't describe it better.",
        "Delete the '[breath]' after 'describe it better'.",
        "Add '[breath]' after 'describe it better'.",
    ),
    (
        "[gasp]请嘉宾面团老师发表一下今天的访谈的感言。",
        "gasp",
        "zh",
        "请嘉宾面团老师发表一下今天的访谈的感言。",
        "删除‘请嘉宾面团’前面的‘[gasp]’",
        "在‘请嘉宾面团’前面加上‘[gasp]’",
    ),
    (
        "煮面时水煮沸溅到了我[laugh]，我的厨艺再一次让我颜面尽失啦[laugh]。",
        "laugh",
        "zh",
        "煮面时水煮沸溅到了我，我的厨艺再一次让我颜面尽失啦。",
        "删除‘溅到了我’后面的‘[laugh]’，删除‘颜面尽失啦’后面的‘[laugh]’",
        "在‘溅到了我’后面加上‘[laugh]’，在‘颜面尽失啦’后面加上‘[laugh]’",
    ),
    (
        "面对那么刁钻的问题，他居然不慌不忙[smack]，一番话说得对方哑口无言，最后只能[clap]给他竖起大拇指。",
        "smack",
        "zh",
        "面对那么刁钻的问题，他居然不慌不忙，一番话说得对方哑口无言，最后只能[clap]给他竖起大拇指。",
        "删除‘不慌不忙’后面的‘[smack]’",
        "在‘不慌不忙’后面加上‘[smack]’",
    ),
    (
        "那模样竟让几人生出一份郁闷[breath]，[breath]似乎一拳打在了棉花上，对方没有半分的反应。",
        "breath",
        "zh",
        "那模样竟让几人生出一份郁闷，似乎一拳打在了棉花上，对方没有半分的反应。",
        "删除‘似乎一拳’前面的‘[breath]’",
        "在‘似乎一拳’前面加上‘[breath]’",
    ),
    (
        "一进了关押了谢必安的屋子，[question-en]？都在呢。",
        "question-en",
        "zh",
        "一进了关押了谢必安的屋子，？都在呢。",
        "删除‘都在呢’前面的‘[question-en]’",
        "在‘都在呢’前面加上‘[question-en]’",
    ),
]


@pytest.mark.parametrize(("transcript", "tag", "lang", "synthetic_text", "forward", "reverse"), RELEASE_ROWS)
def test_record_matches_the_release(
    transcript: str, tag: str, lang: str, synthetic_text: str, forward: str, reverse: str
) -> None:
    record = nv_record(nv_row(transcript, lang), tag)
    assert (record["edited_text"], record["instruction"], record["reverse_instruction"]) == (
        synthetic_text,
        forward,
        reverse,
    )


def test_record_fields() -> None:
    record = nv_record(nv_row("我们[laugh]走吧，[breath]明天[laugh]再来。", "zh", "c1"), "laugh")
    assert record["id"] == "nonverbal/insertion/c1_laugh"
    assert (record["task"], record["subset"], record["edit_type"]) == ("nonverbal", "insertion", "insertion")
    assert record["original_text"] == "我们[laugh]走吧，[breath]明天[laugh]再来。"
    assert record["edited_text"] == "我们走吧，[breath]明天再来。"
    assert record["edits"] == [
        {"op": "deletion", "target": "[laugh]", "new": "", "direction": "", "region": [2, 9]},
        {"op": "deletion", "target": "[laugh]", "new": "", "direction": "", "region": [22, 29]},
    ]
    assert record["num_edits"] == 2
    assert record["meta"] == {"nv_tag": "laugh", "gaps": [2, 6]}
    assert record["instruction"] == "删除‘走吧’前面的‘[laugh]’，删除‘再来’前面的‘[laugh]’"
    assert record["reverse_instruction"] == "在‘走吧’前面加上‘[laugh]’，在‘再来’前面加上‘[laugh]’"


def test_tag_run_gets_one_clause_and_anchor_grows_until_unique() -> None:
    record = nv_record(nv_row("We go home. [laugh] [laugh] We go home now.", "en"), "laugh")
    assert record["num_edits"] == 2
    assert record["instruction"] == "Delete the '[laugh]' before 'We go home now'."


def test_tail_tag_takes_the_left_anchor_and_head_tag_the_right() -> None:
    assert nv_record(nv_row("我们明天再来吧[sigh]", "zh"), "sigh")["instruction"] == "删除‘明天再来吧’后面的‘[sigh]’"
    record = nv_record(nv_row("[sigh]So she said it again.", "en"), "sigh")
    assert record["instruction"] == "Delete the '[sigh]' before 'So she said'."


def test_en_anchor_keeps_accented_words() -> None:
    record = nv_record(nv_row("We all went to the café [laugh]", "en"), "laugh")
    assert record["instruction"] == "Delete the '[laugh]' after 'to the café'."


def test_tag_inside_an_aligned_token_gives_no_record() -> None:
    row = nv_row("价格是21[cough]500元左右", "zh")
    assert make_candidate({**row, "nv_tag": "cough"}, "insertion", None) is None


def test_text_from_asr_is_rejected() -> None:
    row = {**nv_row("我们[laugh]走吧明天再来", "zh"), "text": "我们走吧明天再来。", "nv_tag": "laugh"}
    with pytest.raises(ValueError, match="--use-transcript"):
        make_candidate(row, "insertion", None)


def test_locate_mid_gap_and_fix_duration() -> None:
    align_word = [["a", 0.0, 0.5], ["b", 1.5, 2.0], ["c", 2.0, 2.5]]
    (segment,) = locate_nv_segments(align_word, [1], "laugh", 3.0)
    assert segment == {"start": 0.4, "end": 1.6, "fix_dur": 0.4, "status": "ok", "tag": "laugh"}


def test_locate_narrow_gap_falls_back_and_close_events_merge() -> None:
    align_word = [["a", 0.0, 0.5], ["b", 0.5, 0.6], ["c", 0.6, 1.0], ["d", 1.5, 2.0]]
    segments = locate_nv_segments(align_word, [1, 2], "breath", 2.0)
    # Gap 1 at 0.5 widens to [0.4, 0.55]; gap 2 at 0.6 to [0.55, 0.7]; padded, they overlap and merge.
    assert len(segments) == 1
    (merged,) = segments
    assert merged["tag"] == "breath+breath" and merged["status"] == "fallback" and merged["fix_dur"] == 0.2
    assert merged["start"] == pytest.approx(0.3) and merged["end"] == pytest.approx(0.8)


def test_locate_head_tail_and_unreliable() -> None:
    align_word = [["a", 0.5, 1.0], ["b", 1.0, 1.9]]
    head, tail = locate_nv_segments(align_word, [0, 2], "cough", 2.0)
    assert (head["start"], head["end"], head["fix_dur"], head["status"]) == (0.0, pytest.approx(0.6), 0.15, "ok")
    assert (tail["start"], tail["end"], tail["status"]) == (pytest.approx(1.6), 2.0, "fallback")
    (head,) = locate_nv_segments([["a", 0.0, 0.08], ["b", 0.5, 1.0]], [0], "cough", 2.0)
    assert head["status"] == "unreliable"


def make_nv_record(tmp_path: Path, gaps: list[int]) -> dict:
    """A 2 s, 16 kHz sine clip with words at [0.1, 0.5], [0.9, 1.5], [1.5, 1.9] and a laugh in gap 1."""
    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, 0.2 * np.sin(2 * np.pi * 220 * np.arange(32000) / 16000), 16000)
    return {
        "id": "nonverbal/insertion/u0",
        "task": "nonverbal",
        "language": "en",
        "audio_path": str(audio_path),
        "align_word": [["We", 0.1, 0.5], ["go", 0.9, 1.5], ["home", 1.5, 1.9]],
        "original_text": "We [laugh] go [sigh] home.",
        "edited_text": "We go [sigh] home.",
        "meta": {"nv_tag": "laugh", "gaps": gaps},
    }


def test_synthesize_nonverbal_record(tmp_path: Path) -> None:
    model = FakeCFM()
    row = synthesize_record(make_nv_record(tmp_path, [1]), model, FakeVocos(), str(tmp_path), **SAMPLING)
    assert row["status"] == "ok"
    assert row["nv_segments"] == [{"start": 0.4, "end": 1.0, "fix_dur": 0.2, "status": "ok", "tag": "laugh"}]
    (region,) = row["edit_regions"]
    assert region == {"original_start": 0.4, "original_end": 1.0, "synthetic_start": 0.4, "synthetic_end": 0.6}
    (kwargs,) = model.sample_kwargs
    assert kwargs["text"] == convert_char_to_pinyin(["We go home."])


def test_synthesize_unlocated_nonverbal_record(tmp_path: Path) -> None:
    record = {**make_nv_record(tmp_path, [0]), "align_word": [["We", 0.0, 0.05], ["go", 0.9, 1.5]]}
    row = synthesize_record(record, FakeCFM(), FakeVocos(), str(tmp_path), **SAMPLING)
    assert row["status"] == "unlocated" and row["synthetic_audio"] == ""


class FakeAligner:
    """Stands in for Qwen3ForcedAligner: one 0.2 s item per whitespace-separated word."""

    texts: list[str] = []

    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> "FakeAligner":
        return cls()

    def align(self, audio: list[str], text: list[str], language: list[str]) -> list[ForcedAlignResult]:
        FakeAligner.texts += text
        return [
            ForcedAlignResult([ForcedAlignItem(word, i * 0.2, i * 0.2 + 0.2) for i, word in enumerate(t.split())])
            for t in text
        ]


def test_align_cli_uses_the_transcript_without_asr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"uuid": "u0", "audio_path": "a.wav", "language": "en", "transcript": "Hi [laugh] there."},
        {"uuid": "u1", "audio_path": "b.wav", "language": "en", "transcript": "[laugh]"},
    ]
    write_jsonl(tmp_path / "manifest.jsonl", rows)
    monkeypatch.setattr(align, "Qwen3ASRModel", None)
    monkeypatch.setattr(align, "Qwen3ForcedAligner", FakeAligner)
    monkeypatch.setattr(FakeAligner, "texts", [])
    argv = ["align", "--input", str(tmp_path / "manifest.jsonl"), "--output", str(tmp_path / "out.jsonl")]
    monkeypatch.setattr(sys, "argv", [*argv, "--use-transcript", "--aligner-model", "Qwen/Qwen3-ForcedAligner-0.6B"])
    align.main()
    assert FakeAligner.texts == ["Hi there."]
    (out,) = read_jsonl(str(tmp_path / "out.jsonl"))
    assert out == {**rows[0], "text": "Hi there.", "align_word": [["Hi", 0.0, 0.2], ["there.", 0.2, 0.4]]}


def test_annotate_cli_writes_one_record_per_tag_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        nv_row("我们[laugh]走吧，[breath]明天再来看看。", "zh", "c1"),
        nv_row("我们走吧，明天再来看看。", "zh", "c2"),
    ]
    write_jsonl(tmp_path / "aligned.jsonl", rows)
    argv = ["annotate", "--task", "nonverbal", "--subset", "insertion", "--input", str(tmp_path / "aligned.jsonl")]
    monkeypatch.setattr(sys, "argv", [*argv, "--output", str(tmp_path / "edits.jsonl")])
    cli.main()
    records = list(read_jsonl(str(tmp_path / "edits.jsonl")))
    assert [record["id"] for record in records] == ["nonverbal/insertion/c1_laugh", "nonverbal/insertion/c1_breath"]
    assert records[1]["edited_text"] == "我们[laugh]走吧，明天再来看看。"


def test_annotate_cli_needs_no_llm_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_jsonl(tmp_path / "aligned.jsonl", [nv_row("我们[laugh]走吧，明天再来看看。", "zh")])
    monkeypatch.setattr(cli, "LLMClient", SimpleNamespace)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    argv = ["annotate", "--task", "nonverbal", "--subset", "insertion", "--input", str(tmp_path / "aligned.jsonl")]
    monkeypatch.setattr(sys, "argv", [*argv, "--output", str(tmp_path / "edits.jsonl")])
    cli.main()
    assert len(list(read_jsonl(str(tmp_path / "edits.jsonl")))) == 1


ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "examples" / "mini_nonverbal"
SCRIPT = ROOT / "scripts" / "run_mini_nonverbal.sh"


def test_mini_manifest_clips() -> None:
    rows = list(read_jsonl(str(MINI / "manifest.jsonl")))
    assert [row["language"] for row in rows].count("zh") == 10
    assert [row["language"] for row in rows].count("en") == 10
    assert len({row["uuid"] for row in rows}) == len(rows)
    for row in rows:
        assert list(row) == ["uuid", "audio_path", "language", "transcript"]
        assert row["audio_path"] == f"examples/mini_nonverbal/audio/{row['uuid']}.wav"
        info = sf.info(str(ROOT / row["audio_path"]))
        assert info.channels == 1 and 4.0 <= info.duration <= 10.0
        assert len(split_by_tag(row)) == 1
    assert {path.name for path in (MINI / "audio").iterdir()} == {f"{row['uuid']}.wav" for row in rows}


def test_mini_precomputed_aligned_rows() -> None:
    manifest = list(read_jsonl(str(MINI / "manifest.jsonl")))
    aligned = list(read_jsonl(str(MINI / "aligned.jsonl")))
    assert [{key: row[key] for key in manifest[0]} for row in aligned] == manifest
    for row in aligned:
        assert row["text"] == strip_nv_tags(row["transcript"], row["language"])[0]
        assert gate(row)
        (split,) = split_by_tag(row)
        assert make_candidate(split, "insertion", None) is not None


def test_mini_readme_table_and_license() -> None:
    readme = (MINI / "README.md").read_text(encoding="utf-8")
    table = re.findall(r"^\| (nvs38k_\d{7}) \| (zh|en) \| ([\d.]+) \| (.+) \|$", readme, flags=re.MULTILINE)
    rows = {row["uuid"]: row for row in read_jsonl(str(MINI / "manifest.jsonl"))}
    assert [uuid for uuid, *_ in table] == list(rows)
    for uuid, language, duration, transcript in table:
        assert language == rows[uuid]["language"]
        assert abs(float(duration) - sf.info(str(ROOT / rows[uuid]["audio_path"])).duration) < 0.01
        assert transcript == rows[uuid]["transcript"]
    assert "CC BY-NC 4.0" in readme
    assert "https://huggingface.co/datasets/nonverbalspeech/nonverbalspeech38k" in readme
    data_license = (MINI / "LICENSE").read_text(encoding="utf-8")
    assert "CC BY-NC 4.0" in data_license
    assert "https://creativecommons.org/licenses/by-nc/4.0/legalcode" in data_license


def test_mini_pipeline_script() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    script = SCRIPT.read_text(encoding="utf-8")
    assert re.findall(r"-m (voxcanvas\.[\w.]+)", script) == [
        "voxcanvas.align",
        "voxcanvas.annotate",
        "voxcanvas.synthesize",
        "voxcanvas.verify.asr",
        "voxcanvas.verify.sim",
        "voxcanvas.export",
    ]
    assert "--use-transcript" in script and "--task nonverbal --subset insertion" in script


def test_asr_reference_drops_tags_only_for_nonverbal_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"id": "n0", "task": "nonverbal", "edited_text": "We go [sigh] home.", "synthetic_audio": "a.wav"},
        {"id": "c0", "task": "single_span", "edited_text": "We said [hello] to him.", "synthetic_audio": "b.wav"},
    ]
    write_jsonl(tmp_path / "in.jsonl", [{**row, "status": "ok", "language": "en"} for row in rows])
    monkeypatch.setattr(FakeASRModel, "transcripts", {"a.wav": "We go home.", "b.wav": "We said hello to him."})
    monkeypatch.setattr(asr, "Qwen3ASRModel", FakeASRModel)
    argv = ["asr", "--input", str(tmp_path / "in.jsonl"), "--output", str(tmp_path / "out.jsonl")]
    monkeypatch.setattr(sys, "argv", [*argv, "--model", "Qwen/Qwen3-ASR-1.7B"])
    asr.main()
    assert [row["asr_score"] for row in read_jsonl(str(tmp_path / "out.jsonl"))] == [0.0, 0.0]
