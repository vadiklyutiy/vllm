# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the scheduler-driven heartbeat / lease-renewal system."""

import time
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    HeartbeatInfo,
    NixlConnectorMetadata,
)
from vllm.v1.outputs import KVConnectorOutput

from .utils import create_request, make_nixl_scheduler

_ENGINE_A = "my-engine-id"


def _sched(kv_lease_duration: int = 30):
    return make_nixl_scheduler(heartbeat=True, kv_lease_duration=kv_lease_duration)


def _req(request_id: int = 1):
    return create_request(request_id=request_id, do_remote_prefill=True)


def _worker_stub():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    w = object.__new__(NixlConnectorWorker)
    w._reqs_to_send = {}
    w._lease_extension = 20
    w._reqs_to_process = set()
    w.consumer_notification_counts_by_req = {}
    w.expected_consumer_notifications_by_req = {}
    w.xfer_stats = MagicMock()
    return w


# ===================================================================
# Scheduler: on_new_request
# ===================================================================


def test_on_new_request_tracks_and_groups():
    """Add two reqs to same engine, one to another; verify grouping."""
    s = _sched()
    s.on_new_request(_req(1))
    s.on_new_request(_req(2))

    assert s._heartbeat_by_engine[_ENGINE_A].req_ids == {"prefill-1", "prefill-2"}
    info = s._heartbeat_by_engine[_ENGINE_A]
    assert (info.host, info.port, info.tp_size) == ("my-host", 1234, 1)
    assert s._heartbeat_req_engine["id-1"] == (_ENGINE_A, "prefill-1")

    # Different engine.
    r3 = _req(3)
    r3.kv_transfer_params["remote_engine_id"] = "engine-b"
    s.on_new_request(r3)
    assert len(s._heartbeat_by_engine) == 2


@pytest.mark.parametrize(
    "make_req",
    [
        lambda: create_request(request_id=2, do_remote_decode=True),
        lambda: create_request(request_id=3),  # no kv_transfer_params
    ],
    ids=["decode", "plain"],
)
def test_on_new_request_ignores_non_prefill(make_req):
    s = _sched()
    s.on_new_request(make_req())
    assert len(s._heartbeat_by_engine) == 0


# ===================================================================
# Scheduler: _stop_heartbeat
# ===================================================================


def test_stop_heartbeat_partial_and_full():
    """Stop one of two reqs on same engine, then stop the other."""
    s = _sched()
    s.on_new_request(_req(1))
    s.on_new_request(_req(2))

    s._stop_heartbeat("id-1")
    assert s._heartbeat_by_engine[_ENGINE_A].req_ids == {"prefill-2"}
    assert "id-1" not in s._heartbeat_req_engine

    s._stop_heartbeat("id-2")
    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


# ===================================================================
# Scheduler: build_connector_meta throttling
# ===================================================================


def test_build_connector_meta_heartbeat_throttling():
    # kv_lease_duration=30 => _heartbeat_interval = 30 // 6 = 5
    s = _sched(kv_lease_duration=30)
    s.on_new_request(_req(1))

    # Ensure the first call triggers by placing last_heartbeat far in the past.
    s._last_heartbeat_time = time.perf_counter() - 10
    meta1 = s.build_connector_meta(MagicMock())
    assert _ENGINE_A in meta1.heartbeat_by_engine

    # Immediate second call is throttled (< 5s since last).
    meta2 = s.build_connector_meta(MagicMock())
    assert len(meta2.heartbeat_by_engine) == 0


def test_build_connector_meta_heartbeats_step_with_new_reads():
    """A step that starts a read heartbeats even inside the throttle interval,
    so D reads with P's lease just renewed."""
    s = _sched(kv_lease_duration=30)
    req = _req(1)
    s.on_new_request(req)
    # Another request's heartbeat went out just now.
    s._last_heartbeat_time = time.perf_counter()
    s._reqs_need_recv[req.request_id] = (req, ([10, 11, 12],), (), True)

    meta = s.build_connector_meta(MagicMock())

    assert "id-1" in meta.reqs_to_recv
    assert meta.heartbeat_by_engine[_ENGINE_A].req_ids == {"prefill-1"}


# ===================================================================
# Scheduler: cleanup paths (update_connector_output / request_finished)
# ===================================================================


def test_update_connector_output_stops_heartbeat():
    s = _sched()
    s.on_new_request(_req(1))

    s.update_connector_output(
        KVConnectorOutput(
            finished_sending=None,
            finished_recving={"id-1"},
            invalid_block_ids=set(),
        )
    )

    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


def test_request_finished_stops_heartbeat():
    s = _sched()
    r = _req(1)
    s.on_new_request(r)

    # Simulate update_state_after_alloc having consumed do_remote_prefill.
    r.kv_transfer_params["do_remote_prefill"] = False
    s.request_finished(r, block_ids=())

    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


# ===================================================================
# Worker: _handle_heartbeat
# ===================================================================


def test_handle_heartbeat():
    w = _worker_stub()
    far_future = time.perf_counter() + 99999
    w._reqs_to_send = {"req-a": 100.0, "req-b": far_future}

    before = time.perf_counter()
    w._handle_heartbeat("req-a,req-b,req-unknown")

    # req-a: pushed forward to ~now+20.
    assert w._reqs_to_send["req-a"] >= before + 20
    # req-b: already far out, max() keeps it.
    assert w._reqs_to_send["req-b"] >= far_future
    # req-unknown: not added.
    assert "req-unknown" not in w._reqs_to_send


def test_reaper_reclaims_expired_leases_behind_heartbeated_head(monkeypatch):
    """A heartbeated head entry must not strand expired leases behind it.

    Mirrors the stdlib repro in https://github.com/vllm-project/vllm/issues/58222:
    A is inserted first and kept alive via heartbeats; B and C expire and
    must still be reaped. The old early-exit scan returned nothing here.
    """
    w = _worker_stub()
    w._reqs_to_process.update(("A", "B", "C"))
    # Insertion order is A, B, C. Heartbeats will keep A at the head.
    w._reqs_to_send = {"A": 30.0, "B": 32.0, "C": 34.0}

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker

    now = {"t": 0.0}
    monkeypatch.setattr(base_worker.time, "perf_counter", lambda: now["t"])
    for t in (20.0, 40.0, 60.0, 80.0):
        now["t"] = t
        w._handle_heartbeat("A")

    # A was renewed in place: still the dict head, expiry 100.
    assert next(iter(w._reqs_to_send)) == "A"
    assert w._reqs_to_send["A"] == pytest.approx(100.0)

    done: set[str] = set()
    now["t"] = 50.0
    w._reap_expired_send_leases(done)

    assert done == {"B", "C"}
    assert list(w._reqs_to_send) == ["A"]
    assert w._reqs_to_process == {"A"}
    assert w.xfer_stats.record_kv_expired_req.call_count == 2

    # now == expires is expired (now >= expires): A is due at 100.
    done_eq: set[str] = set()
    now["t"] = 100.0
    w._reap_expired_send_leases(done_eq)
    assert done_eq == {"A"}
    assert not w._reqs_to_send
    assert not w._reqs_to_process
    assert w.xfer_stats.record_kv_expired_req.call_count == 3


def test_reaper_reclaims_when_now_equals_expiry(monkeypatch):
    """The reaper must reclaim on the now == expires boundary."""
    w = _worker_stub()
    w._reqs_to_process.add("eq")
    w._reqs_to_send = {"eq": 50.0}

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker

    monkeypatch.setattr(base_worker.time, "perf_counter", lambda: 50.0)

    done: set[str] = set()
    w._reap_expired_send_leases(done)

    assert done == {"eq"}
    assert not w._reqs_to_send
    assert not w._reqs_to_process
    w.xfer_stats.record_kv_expired_req.assert_called_once()


def test_reaper_reclaims_shorter_lease_behind_later_deadline(monkeypatch):
    """A later-inserted longer TTL must not block an earlier-expiring entry.

    Delete+reinsert on heartbeat cannot restore expiry order in this case
    (issue #58222), so the reaper must scan the whole map.
    """
    w = _worker_stub()
    w._reqs_to_process.update(("X", "Y"))
    w._reqs_to_send = {"X": 55.0, "Y": 46.0}

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker

    monkeypatch.setattr(base_worker.time, "perf_counter", lambda: 50.0)

    done: set[str] = set()
    w._reap_expired_send_leases(done)

    assert done == {"Y"}
    assert list(w._reqs_to_send) == ["X"]
    assert w._reqs_to_process == {"X"}


# ===================================================================
# Worker (D): reads past P's lease
# ===================================================================


@pytest.mark.parametrize(
    ("heartbeat_times", "failed_heartbeat_times", "read_time", "expect_expired"),
    [
        # The request reaches D after P's 30s lease ran out (issue #61229).
        pytest.param([], [], 60.0, True, id="read_after_lease"),
        pytest.param([], [], 10.0, False, id="read_within_lease"),
        # Steady heartbeats from before the deadline keep the blocks held.
        pytest.param(
            list(range(5, 65, 5)), [], 62.0, False, id="renewed_while_waiting"
        ),
        # A first heartbeat late in the lease still renews it.
        pytest.param([26.0], [], 39.0, False, id="renewed_near_lease_end"),
        # P already released the blocks when the first heartbeat arrived.
        pytest.param([40.0, 45.0, 50.0], [], 52.0, True, id="renewals_too_late"),
        # A stall longer than the lease extension lets P release the blocks.
        pytest.param([5.0, 10.0, 15.0, 40.0, 45.0], [], 47.0, True, id="stall"),
        # A heartbeat that did not reach P neither extends nor ends the run.
        pytest.param([20.0, 30.0], [25.0], 44.0, False, id="failed_heartbeat"),
    ],
)
def test_d_declines_read_past_lease(
    monkeypatch, heartbeat_times, failed_heartbeat_times, read_time, expect_expired
):
    """D must not read P's blocks once the lease on them is lost: P may have
    reused them for another request, and the read would succeed silently."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import pull_worker

    w = _worker_stub()  # _lease_extension = 20
    w._lease_expiry_margin = 5.0
    w._renewal_slack = 1.0
    w._lease_renewals = {}
    w._engine_clock_offset = {_ENGINE_A: 0.0}

    now = {"t": 0.0}
    monkeypatch.setattr(pull_worker.time, "perf_counter", lambda: now["t"])

    heartbeat = NixlConnectorMetadata()
    heartbeat.heartbeat_by_engine = {
        _ENGINE_A: HeartbeatInfo(
            req_ids={"prefill-1"}, host="my-host", port=1234, tp_size=1
        )
    }
    for t in sorted(heartbeat_times + failed_heartbeat_times):
        now["t"] = t
        renewed = [] if t in failed_heartbeat_times else [_ENGINE_A]
        w._record_lease_renewals(heartbeat, renewed)

    meta = NixlConnectorMetadata()
    meta.add_new_req_to_recv(
        request_id="id-1",
        local_block_ids=([10, 11],),
        kv_transfer_params={
            "remote_block_ids": ([20, 21],),
            "remote_engine_id": _ENGINE_A,
            "remote_request_id": "prefill-1",
            "remote_host": "my-host",
            "remote_port": 1234,
            # P granted a 30s lease at t=0.
            "remote_blocks_expiry_time": 30.0,
        },
    )
    now["t"] = read_time
    assert w._is_remote_read_expired(meta.reqs_to_recv["id-1"]) is expect_expired
