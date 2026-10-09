# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import requests

from vllm.benchmarks.sweep import server as sweep_server
from vllm.benchmarks.sweep.server import ServerProcess
from vllm.utils.network_utils import get_open_port


class _RunningProcess:
    returncode = None

    def poll(self):
        return None


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _make_server(port: int) -> ServerProcess:
    server = ServerProcess(
        ["vllm", "serve", "--host", "127.0.0.1", "--port", str(port)],
        [],
        show_stdout=False,
    )
    server._server_process = _RunningProcess()
    return server


@pytest.fixture
def health_port(request):
    """Port of a server whose /health answers after `request.param` seconds."""
    delay = request.param

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            time.sleep(delay)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


@pytest.mark.parametrize("health_port", [0], ids=["delay=0s"], indirect=True)
@pytest.mark.parametrize(
    "timeout", [5, sys.maxsize, pytest.param(10**309, id="10**309")]
)
def test_wait_until_ready_healthy(health_port, timeout):
    _make_server(health_port).wait_until_ready(timeout=timeout)


def test_wait_until_ready_crash_before_deadline():
    """A crash in the last retry interval is reported as a crash."""
    server = ServerProcess(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(0.2); raise SystemExit(3)",
            "--host",
            "127.0.0.1",
            "--port",
            str(get_open_port()),
        ],
        [],
        show_stdout=False,
    )
    with server, pytest.raises(RuntimeError, match="return code 3"):
        server.wait_until_ready(timeout=1)


@pytest.mark.parametrize("health_port", [3], ids=["delay=3s"], indirect=True)
def test_wait_until_ready_slow_health_times_out(health_port):
    """A /health probe that outlasts the budget must not block or succeed."""
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        _make_server(health_port).wait_until_ready(timeout=1)
    assert time.monotonic() - start < 2


@pytest.mark.parametrize(
    ("probes", "gives_up_at"),
    [
        # Fails with 0.5s left: the retry sleep must not run past the deadline.
        ([(0.5, False), (0.0, True)], 1.0),
        # Healthy, but only after the deadline.
        ([(1.5, True)], 1.5),
    ],
)
def test_wait_until_ready_no_success_after_deadline(monkeypatch, probes, gives_up_at):
    clock = _FakeClock()
    probe_iter = iter(probes)

    def fake_get(url, **kwargs):
        duration, healthy = next(probe_iter)
        clock.now += duration
        if not healthy:
            raise requests.ConnectionError
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(sweep_server, "time", clock)
    monkeypatch.setattr(sweep_server.requests, "get", fake_get)

    with pytest.raises(TimeoutError):
        _make_server(8000).wait_until_ready(timeout=1)
    assert clock.now == gives_up_at
