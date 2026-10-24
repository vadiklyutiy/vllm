# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from tests.utils import ROCM_EXTRA_ARGS, RemoteOpenAIServer

# glm47 sets a default strict level. Its grammar does not depend on the model,
# so a small chat model is enough.
MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server():
    args = [
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "glm47",
        "--max-model-len",
        "4096",
    ] + ROCM_EXTRA_ARGS
    with RemoteOpenAIServer(MODEL, args) as remote_server:
        yield remote_server


def test_parser_default_keeps_response_format_with_auto_tools(
    server: RemoteOpenAIServer,
):
    """Without a strict tool, the parser default strict level must not let auto
    tool calls bypass the response_format (#62483)."""
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "const": "GRAMMAR_WORKS"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    chat_completion = server.get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 2+2? Use the calculator tool."}],
        temperature=0,
        max_completion_tokens=256,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ],
        tool_choice="auto",
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        },
    )

    message = chat_completion.choices[0].message
    assert not message.tool_calls
    assert json.loads(message.content) == {"answer": "GRAMMAR_WORKS"}
