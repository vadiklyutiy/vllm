# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import anthropic
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SERVED_MODEL_NAME = "claude-3-7-sonnet-latest"

MESSAGES = [{"role": "user", "content": "What is the weather in Paris?"}]
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


@pytest.fixture(scope="module")
def server():
    # No --enable-auto-tool-choice / --tool-call-parser: counting needs neither.
    args = [
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--served-model-name",
        SERVED_MODEL_NAME,
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client_anthropic() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_choice",
    [
        pytest.param(None, id="omitted"),
        pytest.param({"type": "auto"}, id="auto"),
        pytest.param({"type": "any"}, id="any"),
        pytest.param({"type": "tool", "name": "get_weather"}, id="tool"),
    ],
)
async def test_count_tokens_with_tools_needs_no_tool_parser(
    client: anthropic.AsyncAnthropic, tool_choice
):
    """count_tokens generates nothing, so it counts tools for any tool_choice
    even when the server has no tool parser."""
    without_tools = await client.messages.count_tokens(
        model=SERVED_MODEL_NAME, messages=MESSAGES
    )
    tool_choice_none = await client.messages.count_tokens(
        model=SERVED_MODEL_NAME,
        messages=MESSAGES,
        tools=TOOLS,
        tool_choice={"type": "none"},
    )
    kwargs = {} if tool_choice is None else {"tool_choice": tool_choice}
    result = await client.messages.count_tokens(
        model=SERVED_MODEL_NAME, messages=MESSAGES, tools=TOOLS, **kwargs
    )
    assert result.input_tokens == tool_choice_none.input_tokens
    assert result.input_tokens > without_tools.input_tokens
