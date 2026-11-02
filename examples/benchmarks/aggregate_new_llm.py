"""
Aggregate LLM-as-Judge metrics across all scenarios.

Scans ``eval_root`` for per-scenario metric JSONs and produces:
  • ``eval_root/aggregate_metrics.json`` — overall + per-sector summaries
  • ``eval_root/aggregate_metrics_detailed.json`` — per-scenario breakdown

This is a synchronous function (no LLM call).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any


def aggregate_new_llm_metrics(eval_root: str) -> dict[str, Any]:
    """Walk every scenario sub-directory, collect metrics, aggregate.

    Returns the summary dict and saves two JSON files.
    """
    from judge_utils import load_metric

    per_scenario: list[dict[str, Any]] = []

    for entry in sorted(os.listdir(eval_root)):
        scenario_dir = os.path.join(eval_root, entry)
        if not os.path.isdir(scenario_dir):
            continue
        # Only consider directories that contain a spec.json (i.e. scenario outputs)
        spec_path = os.path.join(scenario_dir, "spec.json")
        if not os.path.exists(spec_path):
            continue

        sector = "unknown"
        scenario_id = entry
        with open(spec_path, "r") as f:
            spec = json.load(f)
        sector = spec.get("sector", "unknown")
        scenario_id = str(spec.get("scenario_id", scenario_id))

        composite = load_metric(scenario_dir, "composite")
        da = load_metric(scenario_dir, "da")
        ia = load_metric(scenario_dir, "ia")
        eff = load_metric(scenario_dir, "eff")
        cpv = load_metric(scenario_dir, "cpv")

        if not composite:
            continue  # metrics not computed for this scenario

        per_scenario.append({
            "scenario_id": scenario_id,
            "sector": sector,
            "CM": composite.get("scenario_score", None),
            "DA": da.get("scenario_score", None),
            "IA": ia.get("scenario_score", None),
            "EFF": eff.get("scenario_score", None),
            "CPV": cpv.get("scenario_score", None),
            "agent_composites": composite.get("agent_composites", {}),
        })

    if not per_scenario:
        empty: dict[str, Any] = {"status": "no_scenarios_with_metrics"}
        with open(os.path.join(eval_root, "aggregate_metrics.json"), "w") as f:
            json.dump(empty, f, indent=2)
        return empty

    # ── Aggregate ─────────────────────────────────────────────────────
    def _mean(vals: list[float | None]) -> float | None:
        nums = [v for v in vals if v is not None]
        return round(sum(nums) / len(nums), 4) if nums else None

    metrics = ["CM", "DA", "IA", "EFF", "CPV"]

    overall: dict[str, Any] = {
        "n_scenarios": len(per_scenario),
    }
    for m in metrics:
        overall[f"mean_{m}"] = _mean([s[m] for s in per_scenario])

    # Per-sector
    by_sector: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in per_scenario:
        by_sector[s["sector"]].append(s)

    sector_summary: dict[str, dict[str, Any]] = {}
    for sector, scenarios in sorted(by_sector.items()):
        sector_summary[sector] = {
            "n_scenarios": len(scenarios),
        }
        for m in metrics:
            sector_summary[sector][f"mean_{m}"] = _mean([s[m] for s in scenarios])

    summary: dict[str, Any] = {
        "overall": overall,
        "by_sector": sector_summary,
    }

    # Save
    with open(os.path.join(eval_root, "aggregate_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    detailed: dict[str, Any] = {
        "overall": overall,
        "by_sector": sector_summary,
        "per_scenario": per_scenario,
    }
    with open(os.path.join(eval_root, "aggregate_metrics_detailed.json"), "w") as f:
        json.dump(detailed, f, indent=2)

    return summary
