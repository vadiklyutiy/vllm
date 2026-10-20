# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.parser.engine.replay_harness import (
    DUMMY_TOOLS,
    MockTokenizer,
    _test_request,
    collect_output,
    replay_streaming,
)
from vllm.parser.deepseek_v41 import deepseek_v41_config
from vllm.parser.parser_manager import ParserManager

CALLS = (
    '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="city" string="true">杭州</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="count" string="false">42</｜DSML｜ parameter>\n'
    '</｜DSML｜ invoke>\n<｜DSML｜ invoke name="add">\n'
    '<｜DSML｜ parameter name="x" string="false">1.5</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="y" string="false">2.25</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)


def tokenizer_for(text, special_calls):
    vocab = {"<think>": 50, "</think>": 51}
    if special_calls:
        vocab |= {"<｜DSML｜ calls>": 52, "</｜DSML｜ calls>": 53}
    tokens: list[tuple[int, str]] = []
    while text:
        special = next((marker for marker in vocab if text.startswith(marker)), None)
        piece = special or text[0]
        tokens.append((vocab[piece] if special else 100 + len(tokens), piece))
        text = text[len(piece) :]
    return MockTokenizer(vocab, tokens), tokens


def parser_for(tokenizer, controls):
    cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v41",
        reasoning_parser_name="deepseek_v41",
        enable_auto_tools=True,
    )
    return cls(tokenizer, chat_template_kwargs=controls)


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("special_calls", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_registered_adapters_parse_parallel_calls_across_chunks(
    thinking,
    special_calls,
    chunk_size,
):
    text = ("Plan.</think>" if thinking else "") + "Checking." + CALLS
    tokenizer, tokens = tokenizer_for(text, special_calls)
    parser = parser_for(tokenizer, {"thinking": thinking})
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[50 if thinking else 51],
        )
    )
    assert output.reasoning == ("Plan." if thinking else "")
    assert output.content.strip() == "Checking."
    assert [call["name"] for call in output.tool_calls] == ["get_weather", "add"]
    assert [json.loads(call["arguments"]) for call in output.tool_calls] == [
        {"city": "杭州", "count": 42},
        {"x": 1.5, "y": 2.25},
    ]


@pytest.mark.parametrize("thinking", [False, True])
def test_registered_adapters_parse_complete_output(thinking):
    text = ("Plan.</think>" if thinking else "") + CALLS
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": thinking})
    reasoning, content, calls = parser.parse(
        text,
        _test_request(tools=DUMMY_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert (reasoning or "") == ("Plan." if thinking else "")
    assert not (content or "").strip()
    assert [call.name for call in calls] == ["get_weather", "add"]
    assert json.loads(calls[0].arguments) == {"city": "杭州", "count": 42}


@pytest.mark.parametrize(
    ("controls", "text", "expected_reasoning", "reasoning_tokens"),
    [
        ({"thinking": False}, "12", "", 0),
        ({"enable_thinking": False}, "12", "", 0),
        ({"thinking": True, "reasoning_effort": "none"}, "12", "", 0),
        ({}, "</think>12", "", 0),
        ({}, "Plan.</think>12", "Plan.", 5),
    ],
)
def test_reasoning_adapter_controls_and_usage(
    controls,
    text,
    expected_reasoning,
    reasoning_tokens,
):
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, controls)
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=1,
            finished_on_last=True,
        )
    )
    assert output.reasoning == expected_reasoning
    assert output.content == "12"
    assert (
        parser.reasoning_parser.count_reasoning_tokens([tid for tid, _ in tokens])
        == reasoning_tokens
    )
    assert parser.is_reasoning_end([50, 100, 51])
    assert not parser.is_reasoning_end([51, 100, 50])


def parse_both_ways(text, special_calls=False):
    """Return ``(non_streamed, streamed)`` content and tool calls for *text*."""
    tokenizer, tokens = tokenizer_for(text, special_calls)
    _, content, calls = parser_for(tokenizer, {"thinking": False}).parse(
        text,
        _test_request(tools=DUMMY_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    streamed = collect_output(
        replay_streaming(
            parser_for(tokenizer, {"thinking": False}),
            tokens,
            chunk_size=1,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
        )
    )
    non_streamed = [
        {"name": call.name, "arguments": call.arguments} for call in calls or []
    ]
    return (content or "", non_streamed), (streamed.content, streamed.tool_calls)


INVOKE = '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            INVOKE + '<｜DSML｜ parameter name="city" string="true">杭州',
            {"city": "杭州"},
        ),
        (
            INVOKE + '<｜DSML｜ parameter name="count" string="false">42',
            {"count": 42},
        ),
        (
            INVOKE + '<｜DSML｜ parameter name="city" string="true">杭州\n'
            "</｜DSML｜ invoke>\n</｜DSML｜ calls>",
            {"city": "杭州\n"},
        ),
    ],
    ids=["cut_in_string", "cut_in_number", "no_parameter_close"],
)
def test_unclosed_last_parameter_keeps_valid_arguments(text, expected):
    """The final arguments extend what streaming sent for an unclosed value,
    so they stay valid JSON and match the non-streaming result."""
    (_, non_streamed), (_, streamed) = parse_both_ways(text)
    assert non_streamed == streamed
    assert [call["name"] for call in streamed] == ["get_weather"]
    assert json.loads(streamed[0]["arguments"]) == expected


@pytest.mark.parametrize(
    ("text", "expected_names"),
    [
        ('\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_wea', []),
        (CALLS[: CALLS.index('name="add"') + len('name="ad')], ["get_weather"]),
    ],
    ids=["only_call", "after_complete_call"],
)
def test_output_cut_inside_tool_name_is_not_a_call(text, expected_names):
    (_, non_streamed), (_, streamed) = parse_both_ways(text)
    assert [call["name"] for call in non_streamed] == expected_names
    assert [call["name"] for call in streamed] == expected_names


@pytest.mark.parametrize("special_calls", [False, True])
@pytest.mark.parametrize("after", ["", "\nDone."], ids=["block_only", "text_after"])
def test_block_without_invoke_is_returned_as_content(special_calls, after):
    text = "Hi\n\n<｜DSML｜ calls>\n<url>https://x.com</url>\n</｜DSML｜ calls>" + after
    non_streamed, streamed = parse_both_ways(text, special_calls)
    assert non_streamed == streamed == (text, [])


def test_block_without_invoke_after_tool_call_is_dropped():
    """A tool call still ends the turn when a block without invoke follows."""
    text = CALLS + "\n<｜DSML｜ calls>\njunk\n</｜DSML｜ calls>\nDone."
    (content, non_streamed), (streamed_content, streamed) = parse_both_ways(text)
    assert non_streamed == streamed
    assert [call["name"] for call in streamed] == ["get_weather", "add"]
    assert content.strip() == streamed_content.strip() == ""


def test_python_argument_conversion_and_partial_values():
    converter = deepseek_v41_config().arg_converter
    raw = (
        '<｜DSML｜ parameter name="object" string="false">'
        '{"a": [true, null]}</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="bad" string="false">[broken</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="bare">7</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="text" string="true">  a<b'
    )
    assert json.loads(converter(raw, True)) == {
        "object": {"a": [True, None]},
        "bad": "[broken",
        "bare": 7,
        "text": "  a<b",
    }
