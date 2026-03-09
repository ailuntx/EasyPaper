"""
Thinking-Aware LLM Client
- **Description**:
    - Drop-in replacement for AsyncOpenAI that transparently strips
      thinking / reasoning blocks from model responses.
    - Supports <think>, <thinking>, <reasoning> tag families as well as
      orphaned closing tags (e.g. K2-Think API style).
    - Non-thinking models are unaffected — if no tags are detected the
      original content is returned as-is.
"""

import re
from typing import Optional

from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Thinking-content stripper
# ---------------------------------------------------------------------------

_THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
    re.DOTALL,
)

_ORPHAN_CLOSING_TAGS = ("</think>", "</thinking>", "</reasoning>")


def strip_thinking(text: Optional[str]) -> Optional[str]:
    """
    Remove thinking / reasoning blocks from LLM output.

    - **Description**:
        - Strips matched ``<think>…</think>``, ``<thinking>…</thinking>``,
          and ``<reasoning>…</reasoning>`` blocks (including multi-line).
        - Handles orphaned closing tags where the API omits the opening tag
          (e.g. K2-Think returns ``thought…\\n</think>\\nactual answer``).
        - Returns the original text unchanged when no thinking markers are
          present, so non-thinking models work without side-effects.

    - **Args**:
        - `text` (str | None): Raw LLM response content.

    - **Returns**:
        - `str | None`: Cleaned content with thinking blocks removed.
    """
    if not text:
        return text

    result = _THINKING_BLOCK_RE.sub("", text)

    for tag in _ORPHAN_CLOSING_TAGS:
        if tag in result:
            result = result.split(tag, 1)[-1]

    stripped = result.strip()
    return stripped if stripped else text


# ---------------------------------------------------------------------------
# Transparent AsyncOpenAI wrapper
# ---------------------------------------------------------------------------


class _CompletionsProxy:
    """Intercepts ``chat.completions.create`` to strip thinking content."""

    __slots__ = ("_completions",)

    def __init__(self, completions):
        self._completions = completions

    async def create(self, **kwargs):
        response = await self._completions.create(**kwargs)
        for choice in response.choices:
            if choice.message and choice.message.content:
                choice.message.content = strip_thinking(choice.message.content)
        return response

    def __getattr__(self, name):
        return getattr(self._completions, name)


class _ChatProxy:
    """Proxy that replaces ``chat.completions`` with the stripping variant."""

    __slots__ = ("_chat", "completions")

    def __init__(self, chat):
        self._chat = chat
        self.completions = _CompletionsProxy(chat.completions)

    def __getattr__(self, name):
        return getattr(self._chat, name)


class LLMClient:
    """
    Drop-in replacement for ``AsyncOpenAI``.

    - **Description**:
        - Wraps ``AsyncOpenAI`` and transparently strips thinking /
          reasoning blocks from every ``chat.completions.create`` response.
        - All other attributes and methods are delegated to the inner client.
        - Non-thinking models are unaffected.

    - **Args**:
        - Same keyword arguments as ``AsyncOpenAI``.
    """

    __slots__ = ("_client", "chat")

    def __init__(self, **kwargs):
        self._client = AsyncOpenAI(**kwargs)
        self.chat = _ChatProxy(self._client.chat)

    def __getattr__(self, name):
        return getattr(self._client, name)
