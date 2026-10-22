# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence

import regex as re

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.utils import partial_tag_overlap


class K2HorizonReasoningParser(DeepSeekR1ReasoningParser):
    """Reasoning parser for K2 Horizon."""

    _TOOL_CALLS_START = "<ifm|tool_calls>"
    _EFFORT_TOKENS: dict[str, tuple[str, str]] = {
        "high": ("<ifm|think>", "</ifm|think>"),
        "medium": ("<ifm|think_fast>", "</ifm|think_fast>"),
        "low": ("<ifm|think_faster>", "</ifm|think_faster>"),
    }

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None:
        chat_template_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        effort = chat_template_kwargs.get("reasoning_effort", "high")
        if not isinstance(effort, str) or effort not in self._EFFORT_TOKENS:
            supported = ", ".join(self._EFFORT_TOKENS)
            raise ValueError(
                f"Unsupported reasoning_effort {effort!r}. "
                f"Supported values: {supported}."
            )

        self._start_token, self._end_token = self._EFFORT_TOKENS[effort]
        super().__init__(tokenizer, *args, **kwargs)
        self._tool_calls_start_id = self.vocab.get(self._TOOL_CALLS_START)
        # The model may close its reasoning with the end tag of another effort,
        # e.g. medium and low often end with </ifm|think>.
        self._end_tokens = (
            self.end_token,
            *(end for _, end in self._EFFORT_TOKENS.values()),
        )
        self._end_pattern = re.compile("|".join(map(re.escape, self._end_tokens)))
        self._end_token_ids = {
            self.vocab[end] for end in self._end_tokens if end in self.vocab
        }
        self._stream_buffer = ""
        self._stream_at_start = True
        self._stream_finished = False

    @property
    def start_token(self) -> str:
        return self._start_token

    @property
    def end_token(self) -> str:
        return self._end_token

    def _without_generated_start(self, model_output: str) -> str:
        return model_output.removeprefix(self.start_token)

    def _split_model_output(self, model_output: str) -> tuple[str, str | None]:
        output = self._without_generated_start(model_output)
        tool_start = output.find(self._TOOL_CALLS_START)
        end = self._end_pattern.search(
            output, 0, tool_start if tool_start != -1 else len(output)
        )
        if end is not None:
            return output[: end.start()], output[end.end() :] or None

        if tool_start != -1:
            return output[:tool_start], output[tool_start:]

        return "", output or None

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str, str | None]:
        del request
        return self._split_model_output(model_output)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        for token_id in reversed(input_ids):
            if token_id == self.start_token_id:
                return False
            if token_id in self._end_token_ids or token_id == self._tool_calls_start_id:
                return True
        return False

    def is_reasoning_end_streaming(
        self,
        input_ids: Sequence[int],
        delta_ids: Iterable[int],
    ) -> bool:
        del input_ids
        return self._stream_finished or not self._end_token_ids.isdisjoint(delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        for index, token_id in enumerate(input_ids):
            if token_id in self._end_token_ids:
                return input_ids[index + 1 :]
            if token_id == self._tool_calls_start_id:
                return input_ids[index:]
        return input_ids if self._stream_finished else []

    def _strip_stream_start(self) -> bool:
        if not self._stream_at_start:
            return True
        if self.start_token.startswith(self._stream_buffer):
            if self._stream_buffer != self.start_token:
                return False
            self._stream_buffer = ""
        elif self._stream_buffer.startswith(self.start_token):
            self._stream_buffer = self._stream_buffer[len(self.start_token) :]
        self._stream_at_start = False
        return True

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        del (
            previous_text,
            current_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

        if self._stream_finished:
            return DeltaMessage(content=delta_text) if delta_text else None

        self._stream_buffer += delta_text
        if not self._strip_stream_start():
            return None

        tool_start = self._stream_buffer.find(self._TOOL_CALLS_START)
        explicit_end = self._end_pattern.search(
            self._stream_buffer,
            0,
            tool_start if tool_start != -1 else len(self._stream_buffer),
        )
        if explicit_end is not None:
            reasoning = self._stream_buffer[: explicit_end.start()]
            content = self._stream_buffer[explicit_end.end() :]
        elif tool_start != -1:
            reasoning = self._stream_buffer[:tool_start]
            content = self._stream_buffer[tool_start:]
        else:
            overlap = max(
                partial_tag_overlap(self._stream_buffer, tag)
                for tag in (*self._end_tokens, self._TOOL_CALLS_START)
            )
            sendable_end = len(self._stream_buffer) - overlap
            reasoning = self._stream_buffer[:sendable_end]
            self._stream_buffer = self._stream_buffer[sendable_end:]
            return DeltaMessage(reasoning=reasoning) if reasoning else None

        self._stream_buffer = ""
        self._stream_finished = True
        return DeltaMessage(
            reasoning=reasoning,
            content=content or None,
        )
