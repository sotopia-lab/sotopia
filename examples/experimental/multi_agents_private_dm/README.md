# Multi-Agent Tests

This directory contains test scenarios for Sotopia's multi-agent (3+ agents)
with private action support.

Run the demo script:

```sh
mkdir -p examples/experimental/multi_agents_private_dm/redis-data
redis-stack-server --dir examples/experimental/multi_agents_private_dm/redis-data
uv run examples/experimental/multi_agents_private_dm/multi_agents_private_dm.py
```
