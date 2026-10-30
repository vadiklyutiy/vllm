# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--enable-per-request-metrics`` must reach ``/v1/messages`` responses."""

import anthropic
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"

TIMING_FIELDS = (
    "time_to_first_token_ms",
    "generation_time_ms",
    "queue_time_ms",
    "tokens_per_second",
)


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--enable-per-request-metrics",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client_anthropic() as async_client:
        yield async_client


def _assert_timing_metrics(metrics: object) -> None:
    assert isinstance(metrics, dict)
    assert all(metrics[field] >= 0 for field in TIMING_FIELDS)


@pytest.mark.asyncio
async def test_metrics_non_streaming(client: anthropic.AsyncAnthropic):
    resp = await client.messages.create(
        model=MODEL_NAME,
        max_tokens=4,
        extra_body={"temperature": 0.0},
        messages=[{"role": "user", "content": "Count to four."}],
    )

    _assert_timing_metrics(resp.model_extra.get("metrics"))


@pytest.mark.asyncio
async def test_metrics_on_final_message_delta(client: anthropic.AsyncAnthropic):
    stream = await client.messages.create(
        model=MODEL_NAME,
        max_tokens=4,
        extra_body={"temperature": 0.0},
        messages=[{"role": "user", "content": "Count to four."}],
        stream=True,
    )

    events = [event async for event in stream]
    with_metrics = [e for e in events if "metrics" in (e.model_extra or {})]
    assert [e.type for e in with_metrics] == ["message_delta"]
    _assert_timing_metrics(with_metrics[0].model_extra["metrics"])
