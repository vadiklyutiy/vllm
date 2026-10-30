# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

import pytest
import regex as re

from tests.tool_parsers.utils import run_tool_extraction_streaming
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.functiongemma_tool_parser import FunctionGemmaToolParser


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = {}
    return tokenizer


@pytest.fixture
def parser(mock_tokenizer):
    return FunctionGemmaToolParser(mock_tokenizer)


@pytest.fixture
def mock_request():
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    request.parallel_tool_calls = True
    return request


class TestExtractToolCalls:
    def test_no_tool_calls(self, parser, mock_request):
        model_output = "Hello, how can I help you today?"
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == model_output

    def test_single_tool_call(self, parser, mock_request):
        model_output = (
            "<start_function_call>call:get_weather{location:<escape>London<escape>}"
            "<end_function_call>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert '"location": "London"' in result.tool_calls[0].function.arguments

    def test_multiple_arguments(self, parser, mock_request):
        model_output = (
            "<start_function_call>call:get_weather{"
            "location:<escape>San Francisco<escape>,"
            "unit:<escape>celsius<escape>}"
            "<end_function_call>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = result.tool_calls[0].function.arguments
        assert "San Francisco" in args
        assert "celsius" in args

    def test_text_before_tool_call(self, parser, mock_request):
        model_output = (
            "Let me check the weather for you. "
            "<start_function_call>call:get_weather{location:<escape>Paris<escape>}"
            "<end_function_call>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.content == "Let me check the weather for you."

    def test_multiple_tool_calls(self, parser, mock_request):
        model_output = (
            "<start_function_call>call:get_weather{location:<escape>London<escape>}"
            "<end_function_call>"
            "<start_function_call>call:get_time{timezone:<escape>UTC<escape>}"
            "<end_function_call>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

    def test_non_string_arguments(self, parser, mock_request):
        """The chat template wraps only strings in <escape>; numbers, booleans,
        arrays and objects are written bare and must not be dropped."""
        model_output = (
            "<start_function_call>call:set_alarm{"
            "days:[<escape>mon<escape>,<escape>tue<escape>],enabled:true,hour:7,"
            "label:<escape>Wake up<escape>,snooze:{minutes:5}}"
            "<end_function_call>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "days": ["mon", "tue"],
            "enabled": True,
            "hour": 7,
            "label": "Wake up",
            "snooze": {"minutes": 5},
        }


def _char_deltas(model_output: str) -> list[str]:
    """One character per delta, with special tokens whole."""
    return [
        piece
        for part in re.split(r"(<[a-z_]+>)", model_output)
        for piece in ([part] if part.startswith("<") else part)
    ]


class TestExtractToolCallsStreaming:
    @pytest.mark.parametrize(
        ("args_str", "expected"),
        [
            ("hour:72", {"hour": 72}),
            ("enabled:true", {"enabled": True}),
            (
                "days:[<escape>mon<escape>,<escape>tue<escape>]",
                {"days": ["mon", "tue"]},
            ),
            ("snooze:{minutes:5}", {"snooze": {"minutes": 5}}),
        ],
    )
    def test_non_string_argument(self, parser, mock_request, args_str, expected):
        """Bare values are streamed, and only once complete: ``72`` arrives
        over two deltas and must not be sent as ``7``."""
        model_output = (
            f"<start_function_call>call:set_alarm{{{args_str}}}<end_function_call>"
        )
        reconstructor = run_tool_extraction_streaming(
            parser, _char_deltas(model_output), mock_request
        )

        assert json.loads(reconstructor.tool_calls[0].function.arguments) == expected

    def test_closed_value_sent_when_stream_stops(self, parser, mock_request):
        """A closed string cannot grow, so it is sent without waiting for more
        text, e.g. when the stream stops at ``max_tokens``."""
        model_output = (
            "<start_function_call>call:get_weather{location:<escape>London<escape>"
        )
        reconstructor = run_tool_extraction_streaming(
            parser, _char_deltas(model_output), mock_request
        )

        assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
            "location": "London"
        }


class TestParseArguments:
    def test_empty_arguments(self, parser):
        result = parser._parse_arguments("")
        assert result == {}

    def test_single_string_argument(self, parser):
        result = parser._parse_arguments("city:<escape>Tokyo<escape>")
        assert result == {"city": "Tokyo"}

    def test_multiple_arguments(self, parser):
        args_str = "city:<escape>Tokyo<escape>,country:<escape>Japan<escape>"
        result = parser._parse_arguments(args_str)
        assert result == {"city": "Tokyo", "country": "Japan"}

    def test_numeric_argument(self, parser):
        result = parser._parse_arguments("count:<escape>42<escape>")
        assert result == {"count": 42}

    def test_boolean_argument(self, parser):
        result = parser._parse_arguments("enabled:<escape>true<escape>")
        assert result == {"enabled": True}

    def test_argument_with_spaces(self, parser):
        result = parser._parse_arguments("message:<escape>Hello World<escape>")
        assert result == {"message": "Hello World"}

    def test_none_argument(self, parser):
        """The chat template writes Python ``None`` as bare ``None``."""
        result = parser._parse_arguments("x:None,y:[None,1]")
        assert result == {"x": None, "y": [None, 1]}

    def test_whitespace_before_key(self, parser):
        args_str = "city:<escape>Tokyo<escape>,\nplace:{ country:<escape>Japan<escape>}"
        result = parser._parse_arguments(args_str)
        assert result == {"city": "Tokyo", "place": {"country": "Japan"}}


class TestAdjustRequest:
    def test_skip_special_tokens_disabled(self, parser, mock_request):
        mock_request.tools = [{"type": "function", "function": {"name": "test"}}]
        mock_request.tool_choice = "auto"
        mock_request.skip_special_tokens = True

        result = parser.adjust_request(mock_request)
        assert result.skip_special_tokens is False

    def test_skip_special_tokens_when_tool_choice_none(self, parser, mock_request):
        mock_request.tools = [{"type": "function", "function": {"name": "test"}}]
        mock_request.tool_choice = "none"
        mock_request.skip_special_tokens = True

        result = parser.adjust_request(mock_request)
        assert result.skip_special_tokens is True


class TestBufferDeltaText:
    def test_regular_text_not_buffered(self, parser):
        result = parser._buffer_delta_text("hello")
        assert result == "hello"
        assert parser.buffered_delta_text == ""

    def test_complete_tag_flushed(self, parser):
        parser.buffered_delta_text = "<start_function_"
        result = parser._buffer_delta_text("call>")
        assert "<start_function_call>" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
