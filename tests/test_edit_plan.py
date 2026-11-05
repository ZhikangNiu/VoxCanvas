import random

import pytest
import torch

from voxcanvas.synthesize import FRAMES_PER_SEC, build_adjusted_mel, build_edit_mask, build_edit_plan


# Plans stored in the VoxCanvas-41M synthesis metadata: (lang, edited_text, align_word as "token start end ...",
# source_parts, parts_to_edit, fix_duration). Comments give the subset and the Emilia uuid.
GOLDEN_PLANS = [
    # single_span deletion, ZH_B00000_S02184_W000014
    pytest.param(
        "zh",
        "人活着，开始是半人。",
        "人 0.0 0.32 活 0.32 0.64 着 0.64 0.96 开 1.12 1.44 始 1.44 1.76 是 2.08 2.32 半 2.32 2.64 人 2.64 2.88 "
        "半 2.88 3.12 兽 3.12 3.44",
        [[2.88, 3.44]],
        [[2.88, 2.98]],
        [0.1],
        id="zh-deletion",
    ),
    # single_span insertion, ZH_B00000_S05406_W000033
    pytest.param(
        "zh",
        "然后，马斯克就拿到已经了钱。",
        "然 0.08 0.4 后 0.4 0.64 马 0.96 1.28 斯 1.28 1.44 克 1.44 1.76 就 1.92 2.08 拿 2.08 2.4 到 2.4 2.56 "
        "了 2.56 2.72 钱 2.72 3.2",
        [[2.56, 2.56]],
        [[2.56, 3.184]],
        [0.624],
        id="zh-insertion",
    ),
    # single_span substitution, ZH_B00000_S05817_W000056
    pytest.param(
        "zh",
        "或者从信用社账户中取款。",
        "或 0.0 0.24 者 0.24 0.56 从 0.8 1.12 银 1.12 1.44 行 1.44 1.76 账 1.76 2.08 户 2.08 2.24 中 2.24 2.56 "
        "取 2.64 3.04 款 3.04 3.36",
        [[1.12, 1.76]],
        [[1.12, 2.08]],
        [0.9599999999999999],
        id="zh-substitution",
    ),
    # single_span substitution, two replaced runs meet at a zero-length shared char, ZH_B00000_S03204_W000009
    pytest.param(
        "zh",
        "组合持有一系列积极成长的公司。",
        "组 0.0 0.24 合 0.24 0.48 持 0.48 0.72 有 0.72 0.96 一 0.96 1.2 系 1.2 1.44 列 1.44 1.6 多 1.6 1.92 "
        "元 1.92 2.24 分 2.24 2.48 散 2.48 2.72 的 2.72 2.72 股 2.72 2.88 票 2.96 3.2",
        [[1.6, 3.2]],
        [[1.6, 3.2]],
        [1.6],
        id="zh-substitution-merged",
    ),
    # single_span deletion, EN_B00000_S04583_W000001
    pytest.param(
        "en",
        "I'll again. Perfect.",
        "I'll 0.08 0.16 sit 0.16 0.4 down 1.68 1.84 here 1.84 2.08 again 2.08 2.4 Perfect 2.8 3.36",
        [[0.16, 2.08]],
        [[0.16, 0.26]],
        [0.1],
        id="en-deletion",
    ),
    # single_span insertion (seconds per token clamped to 0.45), EN_B00000_S02761_W000070
    pytest.param(
        "en",
        "They're they have pecan and walnut maple.",
        "They're 0.0 0.48 they 0.48 0.56 have 0.56 0.96 pecan 1.12 1.12 and 1.68 2.0 maple 3.28 3.28",
        [[2.0, 3.28]],
        [[2.0, 2.45]],
        [0.45],
        id="en-insertion",
    ),
    # single_span substitution, EN_B00000_S06059_W000007; the stored plan is [[1.92, 3.2]] because Spider-Man was
    # two tokens there; the aligner keeps it whole.
    pytest.param(
        "en",
        "Oh wow! He's writer for Spider-Man.",
        "Oh 0.0 0.0 wow 0.8 1.2 He's 1.2 1.44 writer 1.44 1.76 for 1.76 1.92 Daredevil 1.92 2.56",
        [[1.92, 2.56]],
        [[1.92, 2.56]],
        [0.6400000000000001],
        id="en-substitution",
    ),
    # single_span substitution, two replaced words meet at a zero-length "and", EN_B00000_S04557_W000027
    pytest.param(
        "en",
        "Ah, she and Mark talking through this.",
        "Ah 0.0 0.48 you 2.16 2.32 and 2.32 2.32 Tally 2.32 2.64 talking 3.04 3.36 through 3.36 3.52 this 3.52 3.84",
        [[2.16, 2.64]],
        [[2.16, 2.66]],
        [0.5000000000000002],
        id="en-substitution-merged",
    ),
    # multi_span, ZH_B00000_S06924_W000021
    pytest.param(
        "zh",
        "啊，亲爱的木主持压轴，给你兜底。",
        "啊 0.0 0.16 木 0.16 0.32 主 0.32 0.48 持 0.48 0.8 压 0.8 0.96 轴 0.96 1.28 哈 1.28 1.6 给 2.24 2.32 "
        "你 2.4 2.4 托 2.4 2.72 底 2.72 2.88",
        [[0.16, 0.16], [1.28, 1.6], [2.4, 2.72]],
        [[0.16, 1.0239999999999998], [2.144, 2.244], [3.0439999999999996, 3.364]],
        [0.8639999999999999, 0.1, 0.3200000000000003],
        id="zh-multi",
    ),
    # multi_span, EN_B00000_S03087_W000147
    pytest.param(
        "en",
        "previous Equation ten a specifically in equation number.",
        "Equation 0.0 1.04 ten 2.48 2.8 a 2.88 3.2 in 4.0 4.0 equation 4.72 4.72 number 5.28 5.84 five 6.32 6.64",
        [[0.0, 0.0], [3.2, 4.0], [6.32, 6.64]],
        [[0.0, 0.45], [3.6500000000000004, 4.1000000000000005], [6.420000000000001, 6.5200000000000005]],
        [0.45, 0.45, 0.1],
        id="en-multi",
    ),
    # location d1 (all occurrences), ZH_B00012_S02287_W000029
    pytest.param(
        "zh",
        "啊，真的是特大！",
        "冤 0.0 0.32 种 0.32 0.56 啊 0.56 1.28 真 1.36 1.6 的 1.6 1.76 是 1.76 2.16 特 2.24 2.8 大 2.8 3.28 "
        "冤 3.28 3.68 种 3.68 3.84",
        [[0.0, 0.56], [3.28, 3.84]],
        [[0.0, 0.1], [2.82, 2.92]],
        [0.1, 0.1],
        id="zh-d1",
    ),
    # location m2 (second occurrence), ZH_B00037_S05936_W000013
    pytest.param(
        "zh",
        "嗯，知道了，没问题。",
        "嗯 0.0 0.16 知 1.6 1.68 道 1.68 1.84 了 1.84 2.08 知 2.72 2.8 道 2.8 2.96 了 2.96 3.2",
        [[2.72, 3.2]],
        [[2.72, 3.2]],
        [0.48],
        id="zh-m2",
    ),
    # location l_i (sentence end), ZH_B00018_S07198_W000000
    pytest.param(
        "zh",
        "羊呢？跑了回来。",
        "羊 0.08 0.32 呢 0.32 0.64 跑 2.48 2.88 了 2.88 3.04",
        [[3.04, 3.04]],
        [[3.04, 3.68]],
        [0.64],
        id="zh-l_i-tail",
    ),
    # location i1 (before every occurrence), EN_B00009_S07528_W000031
    pytest.param(
        "en",
        "Yep. Yep. really That's crazy. really That's crazy. Yeah.",
        "Yep 0.0 0.48 Yep 0.8 1.04 That's 2.0 2.16 crazy 2.48 2.88 That's 3.04 3.2 crazy 3.2 3.68 Yeah 3.68 3.84",
        [[1.04, 2.0], [2.88, 3.04]],
        [[1.04, 1.49], [2.37, 2.8200000000000003]],
        [0.45, 0.45],
        id="en-i1",
    ),
    # location l_t_d (time span), EN_B00000_S02460_W000058
    pytest.param(
        "en",
        "So we changes.",
        "So 0.0 0.32 we 0.4 0.56 can 0.56 0.8 see 0.8 1.36 the 2.16 2.24 changes 2.24 3.12",
        [[0.56, 2.24]],
        [[0.56, 0.66]],
        [0.1],
        id="en-l_t_d",
    ),
    # location l_i (sentence start), EN_B00027_S05079_W000013
    pytest.param(
        "en",
        "Well Oh boy, eh? That was rough.",
        "Oh 0.0 0.24 boy 0.24 0.72 eh 0.72 0.96 That 2.32 2.48 was 2.48 2.88 rough 2.88 3.6",
        [[0.0, 0.0]],
        [[0.0, 0.45]],
        [0.45],
        id="en-l_i-head",
    ),
]


def parse_align(spec: str) -> list[list]:
    words = spec.split()
    return [[words[i], float(words[i + 1]), float(words[i + 2])] for i in range(0, len(words), 3)]


@pytest.mark.parametrize(
    ("lang", "edited_text", "align", "source_parts", "parts_to_edit", "fix_duration"), GOLDEN_PLANS
)
def test_build_edit_plan_golden_plans(
    lang: str,
    edited_text: str,
    align: str,
    source_parts: list[list[float]],
    parts_to_edit: list[list[float]],
    fix_duration: list[float],
) -> None:
    assert build_edit_plan(parse_align(align), edited_text, lang) == (source_parts, parts_to_edit, fix_duration)


@pytest.mark.parametrize(
    ("lang", "align", "edited_text", "plan"),
    [
        # Insertion takes the inter-word gap and (j2 - j1) * seconds per token: 0.9 s / 3 timed tokens.
        ("en", "the 0.0 0.2 red 0.2 0.5 fox 0.5 0.9", "the big red fox", ([[0.2, 0.2]], [[0.2, 0.5]], [0.3])),
        # Fast speech: seconds per token clamps to 0.08, then the 0.18 s floor applies; appended at the last offset.
        ("zh", "我 0.0 0.05 们 0.05 0.1 走 0.1 0.15", "我们走吧啊", ([[0.15, 0.15]], [[0.15, 0.15 + 0.18]], [0.18])),
        # No token with a positive duration: fallback 0.24 s per token.
        ("en", "a 0.0 0.0 b 0.0 0.0", "a c b", ([[0.0, 0.0]], [[0.0, 0.24]], [0.24])),
        # Punctuation-only edits and a curly apostrophe are no spoken change.
        ("en", "the 0.0 0.2 red 0.2 0.5 fox 0.5 0.9", "The red, fox!", ([], [], [])),
        ("en", "its 0.0 0.3 fine 0.3 0.6", "It’s fine.", ([], [], [])),
        # zh splits Latin into letters (the gate only admits single-letter align tokens such as 'B超').
        ("zh", "用 0.0 0.2 a 0.2 0.3 i 0.3 0.4 写 0.4 0.6", "用AI写", ([], [], [])),
        # Repeated words leave the diff a choice of equivalent scripts, and it need not be the one the instruction
        # names: "Remove the first 'it's'" (d4/EN_B00000_S00216_W000012) regenerates the second (2.48 - 2.72).
        (
            "en",
            "Your 0.0 0.16 state 0.16 0.64 and 1.28 1.44 so 1.44 2.0 it's 2.24 2.48 it's 2.48 2.72 like 2.8 3.2",
            "Your state, and so it's like.",
            ([[2.48, 2.72]], [[2.48, 2.58]], [0.1]),
        ),
    ],
)
def test_build_edit_plan_rules(lang: str, align: str, edited_text: str, plan: tuple) -> None:
    assert build_edit_plan(parse_align(align), edited_text, lang) == plan


def test_build_adjusted_mel_swaps_parts_for_zero_frames() -> None:
    # A 3.073 s clip has 289 frames; [2.08, 2.24] = frames [195, 210) becomes round(0.1 * 93.75) = 9 zero frames.
    mel = torch.arange(1.0, 290.0).view(1, 289, 1).expand(1, 289, 100)
    adjusted = build_adjusted_mel(mel, [[2.08, 2.24]], [0.1], 3.073)
    assert adjusted.shape == (1, 283, 100)
    assert torch.equal(adjusted[0, :195, 0], torch.arange(1.0, 196.0))
    assert not adjusted[0, 195:204].any()
    assert torch.equal(adjusted[0, 204:, 0], torch.arange(211.0, 290.0))

    # An insertion keeps every frame; a part past the end is clamped to the audio duration (frame 288 survives).
    adjusted = build_adjusted_mel(mel, [[1.0, 1.0], [3.0, 9.9]], [0.5, 0.1], 3.073)
    assert adjusted.shape == (1, 94 + 47 + (281 - 94) + 9 + 1, 100)
    assert not adjusted[0, 94:141].any() and not adjusted[0, 328:337].any()
    assert adjusted[0, -1, 0].item() == 289.0


def test_build_edit_mask_widens_parts_by_half_a_second_and_clamps() -> None:
    mask = build_edit_mask(300, [[0.2, 0.3], [2.0, 2.1], [3.1, 3.1]])
    expected = torch.ones(1, 300, dtype=torch.bool)
    expected[:, : round(0.8 * FRAMES_PER_SEC)] = False
    expected[:, round(1.5 * FRAMES_PER_SEC) :] = False  # [141, 244) and [244, 338) clamped to 300
    assert torch.equal(mask, expected)


def test_zero_condition_frames_lie_in_the_generated_region() -> None:
    """A zero log-mel frame (magnitude 1) left in the kept condition would be vocoded as a loud artifact."""
    golden = [case.values for case in GOLDEN_PLANS if case.id in ("zh-multi", "en-multi", "zh-d1", "en-i1")]
    cases = [(lang, edited_text, parse_align(align)) for lang, edited_text, align, *_ in golden]
    rng = random.Random(0)
    for _ in range(20):
        align_word, edited, end = [], [], 0.0
        for i in range(60):
            start = round(end + rng.choice([0.0, 0.08, rng.uniform(0.0, 0.6)]), 2)
            end = round(start + rng.choice([0.0, 0.16, rng.uniform(0.0, 0.5)]), 2)
            align_word.append([f"w{i}", start, end])
            op = rng.choices(["keep", "delete", "replace", "insert"], weights=[13, 3, 2, 2])[0]
            new_words = [f"x{rng.randrange(1000)}" for _ in range(rng.randint(1, 3))]
            edited += {"keep": [f"w{i}"], "delete": [], "replace": new_words, "insert": [*new_words, f"w{i}"]}[op]
        cases.append(("en", " ".join(edited), align_word))

    for lang, edited_text, align_word in cases:
        source_parts, parts_to_edit, fix_duration = build_edit_plan(align_word, edited_text, lang)
        audio_duration = align_word[-1][2] + 0.3
        mel = torch.ones(1, int(audio_duration * FRAMES_PER_SEC) + 1, 1)
        cond = build_adjusted_mel(mel, source_parts, fix_duration, audio_duration)
        mask = build_edit_mask(cond.shape[1], parts_to_edit)
        assert source_parts and not ((cond[0, :, 0] == 0) & mask[0]).any(), edited_text
