# Stretcher

Capacity stretching across a fleet of inference nodes. When one node
saturates, Stretcher spreads new work to the **healthiest node in the pool**
instead of queueing behind it.

## Why

A single saturated node queues every new request behind its running batch —
latency climbs even though sibling nodes sit idle. Stretcher reads each
node's `/metrics` gauges (read-only) and routes new work to the least-loaded
node, stretching effective fleet capacity without touching any serving stack.

## Architecture

```
                 ┌──────────────┐
 request ──────► │   NodePool   │
                 │ pick_least_  │──── GET /metrics (read-only) ──► node-a
                 │   loaded()   │──── GET /metrics (read-only) ──► node-b
                 └──────────────┘
```

- **`Node`** — `{name, base_url, weight}`. Health signals scraped from
  Prometheus gauges: `vllm:kv_cache_usage_perc` (or legacy
  `kv_cache_usage_perc`) and queue depth (`vllm:num_requests_waiting` /
  `queue_depth`). Metric names are mapped defensively — vLLM renames them
  across versions.
- **`NodePool.pick_least_loaded()`** — picks the minimum `load_score`
  (KV-cache usage + normalized queue depth, queue dominates); among ties it
  makes a **weighted fair** choice by node weight. Unhealthy nodes are
  skipped; an all-down pool raises instead of silently queueing.
- **Sticky sessions** (optional) — pin a session to its node; re-pins only
  if that node goes unhealthy.

## Usage

```python
from stretcher import Node, NodePool

pool = NodePool(sticky=True)
pool.add(Node("spark-head", "http://192.168.0.182:8000", weight=2.0))
pool.add(Node("spark-worker", "http://192.168.0.198:8000"))

pool.refresh_all()                      # read-only GET /metrics on each
node = pool.pick_least_loaded("tenant-a")
# route the request to node.base_url
```

## Results

- 13 offline tests, all HTTP mocked (`urllib` patched) — zero live network
  in CI.
- Health scoring verified against both current and legacy vLLM metric names.
- Weighted-fair tie-breaking verified statistically (weight 3:1 → ~75%
  share over 400 picks).
- Unhealthy nodes are never picked; sticky sessions survive node failure
  via re-pin.

## Scope & safety

Read-only against serving nodes: only `GET /metrics` is ever called —
never `/v1/*`. No secrets, no config changes, no restarts. Pure stdlib
(`urllib`, `re`, `dataclasses`); pytest is the only dev dependency.

## CI

GitHub Actions: Python 3.11, `pip install pytest`, `pytest -q`.
