# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import itertools
import runpy
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

pytestmark = pytest.mark.cpu_test


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
async def test_response_media_type_matches_payload(monkeypatch, endpoint, stream):
    """Clients must select SSE parsing for streaming completion responses."""
    path = (
        Path(__file__).parents[4]
        / "examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py"
    )
    proxy = runpy.run_path(str(path))["_handle_completions"].__globals__
    app = proxy["app"]
    app.state.ready = asyncio.Event()
    app.state.ready.set()
    app.state.prefill_iterator = itertools.cycle([({}, 0)])
    app.state.decode_iterator = itertools.cycle([0])
    app.state.decode_clients = [{}]
    payload = b'{"choices":[{"text":"hello"}]}'
    body = b"data: " + payload + b"\n\ndata: [DONE]\n\n" if stream else payload

    async def decode_response(*args, **kwargs):
        yield body

    monkeypatch.setitem(proxy, "send_request_to_service", AsyncMock())
    monkeypatch.setitem(proxy, "stream_service_response", decode_response)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        response = await client.post(endpoint, json={"stream": stream})

    expected_type = "text/event-stream" if stream else "application/json"
    assert response.headers["content-type"].split(";")[0] == expected_type
    assert response.content == body


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [200, 500])
async def test_decode_status_and_body_forwarded(monkeypatch, endpoint, stream, status):
    """Decode's status and whole body must reach the client, and an error must
    not come after a 200 status."""
    path = (
        Path(__file__).parents[4]
        / "examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py"
    )
    proxy = runpy.run_path(str(path))["_handle_completions"].__globals__
    app = proxy["app"]
    app.state.ready = asyncio.Event()
    app.state.ready.set()
    prefill_info = {"bootstrap_addr": "http://prefill:8998", "dp_engine_id": {0: "e"}}
    app.state.prefill_iterator = itertools.cycle([(prefill_info, 0)])
    app.state.decode_iterator = itertools.cycle([0])
    body = b'{"message":"from decode"}'
    content_type = (
        "text/event-stream" if status == 200 and stream else "application/json"
    )

    class TwoChunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield body[:5]
            yield body[5:]

    def decode_response(request):
        return httpx.Response(
            status, stream=TwoChunks(), headers={"content-type": content_type}
        )

    monkeypatch.setitem(proxy, "send_request_to_service", AsyncMock())
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(decode_response), base_url="http://decode"
        ) as decode_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client,
    ):
        app.state.decode_clients = [{"client": decode_client}]
        response = await client.post(endpoint, json={"stream": stream})

    assert response.status_code == status
    assert response.headers["content-type"].split(";")[0] == content_type
    assert response.content == body


@pytest.mark.asyncio
@pytest.mark.parametrize("decode_sent_headers", [False, True])
async def test_client_disconnect_closes_decode_request(
    monkeypatch, decode_sent_headers
):
    """A client that leaves before decode's first chunk must cancel the decode
    request. Once decode has sent its headers, the proxy must send its own."""
    path = (
        Path(__file__).parents[4]
        / "examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py"
    )
    proxy = runpy.run_path(str(path))["_handle_completions"].__globals__
    app = proxy["app"]
    app.state.ready = asyncio.Event()
    app.state.ready.set()
    prefill_info = {"bootstrap_addr": "http://prefill:8998", "dp_engine_id": {0: "e"}}
    app.state.prefill_iterator = itertools.cycle([(prefill_info, 0)])
    app.state.decode_iterator = itertools.cycle([0])
    decode_closed = asyncio.Event()
    response_started = asyncio.Event()

    class PendingBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.Event().wait()
            yield b""

        async def aclose(self):
            decode_closed.set()

    async def decode_response(request):
        if decode_sent_headers:
            return httpx.Response(200, stream=PendingBody())
        try:
            await asyncio.Event().wait()
        finally:
            decode_closed.set()

    monkeypatch.setitem(proxy, "send_request_to_service", AsyncMock())
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": b'{"stream": true}'}
        if decode_sent_headers:
            await response_started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            response_started.set()

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/completions",
        "headers": [],
        "query_string": b"",
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(decode_response), base_url="http://decode"
    ) as decode_client:
        app.state.decode_clients = [{"client": decode_client}]
        await asyncio.wait_for(app(scope, receive, send), timeout=5)
        await asyncio.wait_for(decode_closed.wait(), timeout=5)
