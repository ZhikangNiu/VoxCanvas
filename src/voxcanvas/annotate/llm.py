import logging
import re
from collections.abc import Callable

import openai

from voxcanvas.annotate.prompts import GLOBAL_SUBSTITUTION_PROMPT, LANGUAGE_HINT, LOCATION_INSERTION_PROMPT, length_hint
from voxcanvas.text import apply_edits, edit_round_trips, is_valid_insertion, is_valid_replacement


logger = logging.getLogger(__name__)

# Closed reasoning blocks and an unclosed (truncated) trailing one.
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
EDGE_RE = re.compile(r"""^[\s'"‘’“”『』「」]+|[\s'"‘’“”『』「」。.]+$""")
LABEL_RE = re.compile(r"^(?:answer|replacement|insertion|insert|output|输出|答案)\s*[:：]\s*", re.IGNORECASE)
DIRECTION_ZH = {"before": "前面", "after": "后面"}
SPLICE_ORDER = {"before": "your phrase, then `target`", "after": "`target`, then your phrase"}
MAX_DRAWS = 5
# 401 bad key, 402 empty account (deepseek "Insufficient Balance"), 403 no permission, 404 unknown model.
FATAL_STATUS_CODES = (401, 402, 403, 404)


class LLMClient:
    """OpenAI-compatible chat client: one system and one user message per request."""

    def __init__(self, base_url: str, model: str, api_key: str, extra_body: dict | None) -> None:
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=120.0, max_retries=3)
        self.model = model
        self.extra_body = extra_body
        # Incremented without a lock from worker threads; exact only as zero / non-zero.
        self.num_requests = self.num_replies = 0

    def complete(self, prompt: str, system: str) -> str:
        """Reply text without <think> blocks ("" if the reply was only reasoning)."""
        self.num_requests += 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=1.0,
            top_p=0.95,
            max_tokens=256,
            extra_body=self.extra_body,
        )
        self.num_replies += 1
        return THINK_RE.sub("", response.choices[0].message.content or "").strip()


def draw_answer(llm: LLMClient, prompt: str, lang: str, accept: Callable[[str], bool]) -> str | None:
    """Ask up to MAX_DRAWS times; return the first answer (last non-empty reply line) that `accept` approves."""
    for _ in range(MAX_DRAWS):
        try:
            reply = llm.complete(prompt, system=LANGUAGE_HINT[lang])
        except openai.APIError as error:
            # These fail identically on every draw; stop the run instead of redrawing.
            if isinstance(error, openai.APIStatusError) and error.status_code in FATAL_STATUS_CODES:
                raise
            logger.warning(f"LLM request failed after retries: {error}")
            continue
        lines = [line.strip() for line in reply.splitlines() if line.strip()]
        answer = EDGE_RE.sub("", LABEL_RE.sub("", lines[-1])).strip() if lines else ""
        if answer and accept(answer):
            return answer
    return None


def generate_replacement(
    llm: LLMClient,
    raw_text: str,
    target: str,
    lang: str,
    prompt: str,
    accept: Callable[[str], bool],
    target_length: int | None = None,
) -> str | None:
    """New text replacing `target` that passes is_valid_replacement and `accept`; None if every draw fails."""
    text = prompt.format(raw_text=raw_text, target=target, length_hint=length_hint(target_length, lang))
    return draw_answer(llm, text, lang, lambda new: is_valid_replacement(new, target, raw_text, lang) and accept(new))


def generate_insertion(
    llm: LLMClient,
    raw_text: str,
    target: str,
    direction: str,
    lang: str,
    prompt: str,
    accept: Callable[[str], bool],
    target_length: int | None = None,
) -> str | None:
    """Text to insert before/after `target` that passes is_valid_insertion and `accept`; None if every draw fails."""
    text = prompt.format(
        raw_text=raw_text,
        target=target,
        direction=direction,
        direction_upper=direction.upper(),
        direction_zh=DIRECTION_ZH[direction],
        splice_order=SPLICE_ORDER[direction],
        length_hint=length_hint(target_length, lang),
    )
    return draw_answer(
        llm,
        text,
        lang,
        lambda new: is_valid_insertion(new, target, raw_text, direction, lang) and accept(new),
    )


def complete_edits(llm: LLMClient, text: str, edits: list[dict], target: str, lang: str) -> list[dict] | None:
    """Fill the shared `new` of location substitution / insertion edits with LLM text; None if every draw fails.

    A draw is accepted when the edits change the text, add no adjacent duplicate and are undone content-wise by
    replacing / deleting the new text alone.
    """
    op, direction = edits[0]["op"], edits[0]["direction"]

    def accept(new: str) -> bool:
        edited = apply_edits(text, [{**edit, "new": new} for edit in edits], lang)
        return edit_round_trips(text, edited, target, new, op, direction, lang)

    if op == "substitution":
        new = generate_replacement(llm, text, target, lang, GLOBAL_SUBSTITUTION_PROMPT, accept=accept)
    else:
        new = generate_insertion(llm, text, target, direction, lang, LOCATION_INSERTION_PROMPT, accept=accept)
    return None if new is None else [{**edit, "new": new} for edit in edits]
