"""Tests for Stretcher — all HTTP is mocked, no live network."""

import pytest

import stretcher
from stretcher import Node, NodePool, StickySessionError


METRICS_LOADED = """
# HELP vllm:kv_cache_usage_perc KV cache usage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc 0.92
vllm:num_requests_waiting 7
"""

METRICS_HEALTHY = """
vllm:kv_cache_usage_perc 0.10
vllm:num_requests_waiting 0
"""

METRICS_LEGACY = """
kv_cache_usage_perc 0.30
queue_depth 1
"""


class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def mock_urlopen(monkeypatch, bodies):
    """bodies: dict base_url -> body string. Raises for unknown URLs."""
    calls = []

    def fake(url, timeout=None):
        calls.append(url)
        for base, body in bodies.items():
            if url.rstrip("/") == base.rstrip("/") + "/metrics":
                return FakeResponse(body.encode())
        raise OSError(f"no mock for {url}")

    monkeypatch.setattr(stretcher.urllib.request, "urlopen", fake)
    return calls


def test_parse_gauges_defensively():
    node = Node("a", "http://a")
    node._parse(METRICS_LOADED)
    assert node.kv_cache_usage == 0.92
    assert node.queue_depth == 7
    assert node.healthy is True


def test_parse_legacy_metric_names():
    node = Node("a", "http://a")
    node._parse(METRICS_LEGACY)
    assert node.kv_cache_usage == 0.30
    assert node.queue_depth == 1


def test_refresh_marks_unhealthy_on_error(monkeypatch):
    def boom(url, timeout=None):
        raise OSError("down")

    monkeypatch.setattr(stretcher.urllib.request, "urlopen", boom)
    node = Node("a", "http://a")
    node.refresh()
    assert node.healthy is False
    assert node.kv_cache_usage is None


def test_refresh_hits_metrics_path_only(monkeypatch):
    calls = mock_urlopen(monkeypatch, {"http://a:8000": METRICS_HEALTHY})
    node = Node("a", "http://a:8000")
    node.refresh()
    assert calls == ["http://a:8000/metrics"]
    assert node.kv_cache_usage == 0.10


def test_pick_least_loaded_prefers_healthy():
    pool = NodePool()
    loaded = Node("loaded", "http://l")
    loaded.kv_cache_usage = 0.95
    loaded.queue_depth = 8
    free = Node("free", "http://f")
    free.kv_cache_usage = 0.05
    free.queue_depth = 0
    pool.add(loaded)
    pool.add(free)
    picks = {pool.pick_least_loaded().name for _ in range(20)}
    assert picks == {"free"}


def test_unhealthy_node_never_picked():
    pool = NodePool()
    dead = Node("dead", "http://d")
    dead.healthy = False
    alive = Node("alive", "http://a")
    alive.kv_cache_usage = 0.5
    pool.add(dead)
    pool.add(alive)
    for _ in range(10):
        assert pool.pick_least_loaded().name == "alive"


def test_empty_pool_raises():
    with pytest.raises(ValueError):
        NodePool().pick_least_loaded()


def test_all_unhealthy_raises():
    pool = NodePool()
    pool.add(Node("dead", "http://d", ))
    pool.nodes[0].healthy = False
    with pytest.raises(RuntimeError):
        pool.pick_least_loaded()


def test_weighted_fair_among_ties():
    pool = NodePool()
    for name in ("a", "b"):
        n = Node(name, f"http://{name}")
        n.kv_cache_usage = 0.5
        pool.add(n)
    pool.nodes[1].weight = 3.0  # b should win ~75% of the time
    picks = [pool.pick_least_loaded().name for _ in range(400)]
    b_share = picks.count("b") / len(picks)
    assert 0.6 < b_share < 0.9


def test_sticky_session_returns_same_node():
    pool = NodePool(sticky=True)
    n1 = Node("a", "http://a")
    n1.kv_cache_usage = 0.1
    n2 = Node("b", "http://b")
    n2.kv_cache_usage = 0.9
    pool.add(n1)
    pool.add(n2)
    first = pool.pick_least_loaded("sess-1")
    for _ in range(10):
        assert pool.pick_least_loaded("sess-1") is first


def test_sticky_repins_when_node_dies():
    pool = NodePool(sticky=True)
    n1 = Node("a", "http://a")
    n1.kv_cache_usage = 0.1
    n2 = Node("b", "http://b")
    n2.kv_cache_usage = 0.9
    pool.add(n1)
    pool.add(n2)
    assert pool.pick_least_loaded("s").name == "a"
    n1.healthy = False
    assert pool.pick_least_loaded("s").name == "b"


def test_pin_unknown_node_raises():
    pool = NodePool()
    pool.add(Node("a", "http://a"))
    with pytest.raises(StickySessionError):
        pool.pin("s", "nope")


def test_refresh_all_updates_every_node(monkeypatch):
    mock_urlopen(
        monkeypatch,
        {
            "http://a": METRICS_HEALTHY,
            "http://b": METRICS_LOADED,
        },
    )
    pool = NodePool()
    pool.add(Node("a", "http://a"))
    pool.add(Node("b", "http://b"))
    pool.refresh_all()
    assert pool.nodes[0].kv_cache_usage == 0.10
    assert pool.nodes[1].queue_depth == 7
