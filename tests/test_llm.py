from types import SimpleNamespace

import openai
import pytest
from conftest import ScriptedLLM

from voxcanvas.annotate.llm import MAX_DRAWS, LLMClient, generate_insertion, generate_replacement
from voxcanvas.annotate.prompts import (
    INSERTION_PROMPT,
    LANGUAGE_HINT,
    LOCATION_INSERTION_PROMPT,
    SUBSTITUTION_PROMPT,
    length_hint,
)


def accept_any(new: str) -> bool:
    return True


def test_generate_replacement_takes_last_line_and_strips_edges() -> None:
    llm = ScriptedLLM(["Let me think.\n“上海”。"])
    accepted = generate_replacement(
        llm, "我住在北京", "北京", "zh", SUBSTITUTION_PROMPT, accept=accept_any, target_length=2
    )
    assert accepted == "上海"
    assert "raw_text: 我住在北京\ntarget: 北京" in llm.prompts[0]
    assert length_hint(2, "zh") in llm.prompts[0]
    assert llm.systems == [LANGUAGE_HINT["zh"]]


def test_generate_replacement_redraws_until_valid_and_accepted() -> None:
    llm = ScriptedLLM(["北京", "住在", "Answer: 天津", "广州"])
    accepted = generate_replacement(
        llm, "他住在北京", "北京", "zh", SUBSTITUTION_PROMPT, accept=lambda new: new != "天津"
    )
    assert accepted == "广州"
    assert len(llm.prompts) == 4


def test_generate_replacement_gives_up_after_max_draws() -> None:
    llm = ScriptedLLM(["北京"] * MAX_DRAWS)
    assert generate_replacement(llm, "我住在北京", "北京", "zh", SUBSTITUTION_PROMPT, accept_any) is None
    assert len(llm.prompts) == MAX_DRAWS


def test_network_errors_count_as_failed_draws_but_auth_errors_raise() -> None:
    llm = ScriptedLLM([openai.APIConnectionError(request=None), "quickly"])
    assert generate_insertion(llm, "He ran home.", "ran", "after", "en", INSERTION_PROMPT, accept_any) == "quickly"
    response = SimpleNamespace(request=None, status_code=401, headers={})
    llm = ScriptedLLM([openai.AuthenticationError("bad key", response=response, body=None)])
    with pytest.raises(openai.AuthenticationError):
        generate_insertion(llm, "He ran home.", "ran", "after", "en", INSERTION_PROMPT, accept_any)
    response = SimpleNamespace(request=None, status_code=402, headers={})
    llm = ScriptedLLM([openai.APIStatusError("Insufficient Balance", response=response, body=None)])
    with pytest.raises(openai.APIStatusError):
        generate_insertion(llm, "He ran home.", "ran", "after", "en", INSERTION_PROMPT, accept_any)
    response = SimpleNamespace(request=None, status_code=503, headers={})
    llm = ScriptedLLM([openai.APIStatusError("overloaded", response=response, body=None), "quickly"])
    assert generate_insertion(llm, "He ran home.", "ran", "after", "en", INSERTION_PROMPT, accept_any) == "quickly"


@pytest.mark.parametrize(
    ("direction", "zh", "order"),
    [("before", "前面", "your phrase, then `target`"), ("after", "后面", "`target`, then your phrase")],
)
def test_generate_insertion_fills_direction_into_insertion_prompt(direction: str, zh: str, order: str) -> None:
    llm = ScriptedLLM(["quickly"])
    assert generate_insertion(llm, "He ran home.", "ran", direction, "en", INSERTION_PROMPT, accept_any, 1) == "quickly"
    prompt = llm.prompts[0]
    assert f"inserted immediately {direction.upper()} the `target` (加在 target 的{zh})." in prompt
    assert f"in that exact position ({order})." in prompt
    assert f"not already {direction} `target`" in prompt and f"在‘北京’{zh}加上‘非常’" in prompt
    assert length_hint(1, "en") in prompt


def test_generate_insertion_fills_location_prompt_and_rejects_text_already_next_to_the_anchor() -> None:
    llm = ScriptedLLM(["the", "really"])
    raw = "We went to the store and the park."
    inserted = generate_insertion(llm, raw, "store", "before", "en", LOCATION_INSERTION_PROMPT, accept_any)
    assert inserted == "really"
    assert "direction: before" in llm.prompts[0]
    assert "加在 target 的前面" in llm.prompts[0]


def test_llm_client_strips_think_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LLMClient("http://localhost:1/v1", "model", "key", extra_body={"top_k": 20})
    sent = {}
    replies = ["<think>reasoning</think>\n上海", "<think>never closed", None]

    def create(**kwargs: object) -> SimpleNamespace:
        sent.update(kwargs)
        message = SimpleNamespace(content=replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(client.client.chat.completions, "create", create)
    assert client.complete("prompt", system="Answer in English.") == "上海"
    assert client.complete("prompt", system="Answer in English.") == ""
    assert client.complete("prompt", system="Answer in English.") == ""
    assert sent["messages"] == [
        {"role": "system", "content": "Answer in English."},
        {"role": "user", "content": "prompt"},
    ]
    assert sent["extra_body"] == {"top_k": 20}
    assert sent["temperature"] == 1.0 and sent["top_p"] == 0.95 and sent["max_tokens"] == 256
