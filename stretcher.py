"""Stretcher — capacity stretching across a fleet of inference nodes.

When one node saturates, Stretcher spreads new work to the healthiest
node in the pool instead of queueing. Health is scored from Prometheus
gauges exposed on each node's /metrics endpoint (read-only).

Pure stdlib: urllib for HTTP, no third-party runtime deps.
"""

from __future__ import annotations

import random
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = ["Node", "NodePool", "StickySessionError"]

# Prometheus exposition-format line: name{labels} value [timestamp]
_METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"  # metric name
    r"(?:\{(?P<labels>[^}]*)\})?"  # optional labels
    r"\s+(?P<value>[^\s]+)"  # value
)

# Metric names we score on, mapped defensively (vLLM renames across versions).
KV_CACHE_KEYS = (
    "vllm:kv_cache_usage_perc",
    "kv_cache_usage_perc",
    "gpu_cache_usage_perc",
)
QUEUE_DEPTH_KEYS = (
    "vllm:num_requests_waiting",
    "num_requests_waiting",
    "queue_depth",
)


@dataclass
class Node:
    """A single inference node in the pool."""

    name: str
    base_url: str
    weight: float = 1.0
    # Last observed health signals (populated by refresh()).
    kv_cache_usage: Optional[float] = None
    queue_depth: Optional[float] = None
    healthy: bool = True

    @property
    def load_score(self) -> float:
        """Higher = more loaded. Unknown signals count as fully loaded."""
        kv = self.kv_cache_usage if self.kv_cache_usage is not None else 1.0
        q = self.queue_depth if self.queue_depth is not None else 0.0
        # Queue depth dominates: a deep queue is worse than a warm cache.
        return kv + min(q, 10.0) / 10.0

    def refresh(self, timeout: float = 2.0) -> None:
        """GET {base_url}/metrics (read-only) and update health signals."""
        url = self.base_url.rstrip("/") + "/metrics"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            self._parse(body)
            self.healthy = True
        except Exception:
            self.healthy = False
            self.kv_cache_usage = None
            self.queue_depth = None

    def _parse(self, body: str) -> None:
        gauges: Dict[str, float] = {}
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _METRIC_RE.match(line)
            if not m:
                continue
            try:
                gauges[m.group("name")] = float(m.group("value"))
            except ValueError:
                continue
        for key in KV_CACHE_KEYS:
            if key in gauges:
                self.kv_cache_usage = gauges[key]
                break
        for key in QUEUE_DEPTH_KEYS:
            if key in gauges:
                self.queue_depth = gauges[key]
                break


@dataclass
class StickySessionError(KeyError):
    """Raised when a sticky session references an unknown node."""

    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)


@dataclass
class NodePool:
    """A pool of nodes with weighted least-loaded selection."""

    nodes: List[Node] = field(default_factory=list)
    sticky: bool = False
    _sessions: Dict[str, str] = field(default_factory=dict)

    def add(self, node: Node) -> None:
        self.nodes.append(node)

    def refresh_all(self, timeout: float = 2.0) -> None:
        for node in self.nodes:
            node.refresh(timeout=timeout)

    def pick_least_loaded(self, session_id: Optional[str] = None) -> Node:
        """Pick the healthiest node, weighted-fair among ties.

        With sticky=True and a known session_id, returns the node the
        session is pinned to (re-pinning only if that node went unhealthy).
        """
        if not self.nodes:
            raise ValueError("empty node pool")

        if self.sticky and session_id is not None:
            pinned = self._sessions.get(session_id)
            if pinned is not None:
                for node in self.nodes:
                    if node.name == pinned and node.healthy:
                        return node

        live = [n for n in self.nodes if n.healthy]
        if not live:
            raise RuntimeError("no healthy nodes in pool")

        best = min(n.load_score for n in live)
        candidates = [n for n in live if n.load_score <= best + 1e-9]
        # Weighted fair choice among equally-loaded nodes.
        total = sum(max(n.weight, 0.0) for n in candidates) or 1.0
        pick = random.uniform(0.0, total)
        acc = 0.0
        chosen = candidates[0]
        for n in candidates:
            acc += max(n.weight, 0.0)
            if pick <= acc:
                chosen = n
                break

        if self.sticky and session_id is not None:
            self._sessions[session_id] = chosen.name
        return chosen

    def pin(self, session_id: str, node_name: str) -> None:
        if node_name not in {n.name for n in self.nodes}:
            raise StickySessionError(node_name)
        self._sessions[session_id] = node_name
