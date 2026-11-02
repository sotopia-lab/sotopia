"""
Shared utilities for LLM-as-Judge metric evaluation.

Provides helpers used by every metric module:
  - Reading the judge-friendly transcript from disk
  - Building agent / scenario context blocks from the ScenarioSpec
  - Calling the LLM judge with structured (Pydantic) output + retry
  - Persisting metric results as JSON
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("judge")

# ── Filesystem helpers ────────────────────────────────────────────────

def read_transcript(scenario_dir: str) -> str:
    """Return the full text of transcript_pretty.txt (judge-friendly format)."""
    path = os.path.join(scenario_dir, "transcript_pretty.txt")
    with open(path, "r") as f:
        return f.read()


def save_metric(scenario_dir: str, metric_name: str, data: dict[str, Any]) -> str:
    """Write *data* as JSON to ``scenario_dir/metrics/<metric_name>.json``.

    Returns the path written.
    """
    metrics_dir = os.path.join(scenario_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, f"{metric_name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_metric(scenario_dir: str, metric_name: str) -> dict[str, Any]:
    """Load a previously-saved metric JSON.  Returns ``{}`` if missing."""
    path = os.path.join(scenario_dir, "metrics", f"{metric_name}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


# ── Context builders ─────────────────────────────────────────────────

def get_agent_names(spec: dict[str, Any]) -> list[str]:
    """Ordered list of agent role names."""
    return [a["role"] for a in spec["agents"]]


def build_agent_summary(spec: dict[str, Any]) -> str:
    """Return a structured text summary of every agent's knowledge & policies.

    Includes pre-interaction knowledge (with sensitivity), sharing policy,
    and post-interaction knowledge goals.
    """
    lines: list[str] = []
    for agent in spec["agents"]:
        lines.append(f"=== Agent {agent['agent_id']}: {agent['role']} ===")

        # Pre-interaction knowledge
        lines.append("Pre-interaction knowledge:")
        for item_name, item_data in agent["pre_interaction_knowledge"].items():
            sens = item_data.get("sensitivity", "unspecified")
            val = item_data.get("value", "")
            lines.append(f"  - \"{item_name}\" [sensitivity={sens}]: {val}")

        # Sharing policy
        sharing = agent["goals"]["sharing_policy"]
        lines.append("Sharing policy:")
        lines.append("  what_to_share:")
        for item in sharing["what_to_share"]:
            # Look up sensitivity from pre-interaction knowledge
            sens = _lookup_sensitivity(agent, item)
            lines.append(f"    - \"{item}\" [sensitivity={sens}]")
        lines.append("  what_not_to_share:")
        for item in sharing["what_not_to_share"]:
            sens = _lookup_sensitivity(agent, item)
            lines.append(f"    - \"{item}\" [sensitivity={sens}]")

        # Post-interaction knowledge
        post = agent["post_interaction_knowledge"]
        lines.append("Post-interaction knowledge goals:")
        lines.append("  desired_knowledge:")
        for item in post["desired_knowledge"]:
            lines.append(f"    - \"{item}\"")
        lines.append("  cannot_know_knowledge:")
        for item in post["cannot_know_knowledge"]:
            lines.append(f"    - \"{item}\"")

        lines.append("")
    return "\n".join(lines)


def _lookup_sensitivity(agent: dict[str, Any], item_name: str) -> str:
    """Find the sensitivity tag for *item_name* in the agent's pre-interaction knowledge."""
    pre = agent.get("pre_interaction_knowledge", {})
    if item_name in pre:
        return pre[item_name].get("sensitivity", "unspecified")
    # Fuzzy fallback — item name might be a substring of a key
    for key, data in pre.items():
        if item_name.lower() in key.lower() or key.lower() in item_name.lower():
            return data.get("sensitivity", "unspecified")
    return "unspecified"


def build_knowledge_domain_map_text(spec: dict[str, Any]) -> str:
    """Human-readable rendering of the knowledge-domain map."""
    kdm = spec.get("knowledge_domain_map", {})
    if not kdm:
        return "No knowledge domain map provided."
    lines = ["Knowledge Domain Map (which agent owns which knowledge area):"]
    for domain, owner in kdm.items():
        if isinstance(owner, list):
            lines.append(f"  {domain}: {', '.join(owner)}")
        else:
            lines.append(f"  {domain}: {owner}")
    return "\n".join(lines)


def build_knowledge_ownership_table(spec: dict[str, Any]) -> str:
    """Flat table: for every pre-interaction-knowledge item, who owns it + sensitivity."""
    lines = ["Knowledge Ownership (item → agent, sensitivity):"]
    for agent in spec["agents"]:
        role = agent["role"]
        for item_name, item_data in agent["pre_interaction_knowledge"].items():
            sens = item_data.get("sensitivity", "unspecified")
            lines.append(f"  \"{item_name}\" → {role} [sensitivity={sens}]")
    return "\n".join(lines)


def build_context_block(spec: dict[str, Any], scenario_dir: str) -> str:
    """Common context block shared by all metric prompts.

    Contains: scenario goal, agent summaries, knowledge domain map,
    knowledge ownership table, and the full transcript.
    """
    parts = [
        f"## Scenario\n{spec['scenario_goal']}\n",
        f"## Agent Policies\n{build_agent_summary(spec)}\n",
        f"## {build_knowledge_domain_map_text(spec)}\n",
        f"## {build_knowledge_ownership_table(spec)}\n",
        f"## Transcript\n{read_transcript(scenario_dir)}",
    ]
    return "\n".join(parts)


def build_item_checklist(
    spec: dict[str, Any],
    list_key: str,
    parent_key: str = "goals",
    subkey: str | None = "sharing_policy",
) -> str:
    """Build an explicit per-agent item checklist to include in the user prompt.

    *list_key* selects a list within the agent spec, e.g. "what_to_share".
    *parent_key*/*subkey* locate the list:
      goals.sharing_policy.what_to_share  → parent_key="goals", subkey="sharing_policy"
      post_interaction_knowledge.desired_knowledge → parent_key="post_interaction_knowledge", subkey=None
    """
    lines: list[str] = []
    for agent in spec["agents"]:
        role = agent["role"]
        container = agent[parent_key]
        if subkey:
            container = container[subkey]
        items = container[list_key]
        lines.append(f"**{role}** — {list_key} ({len(items)} items):")
        for i, item in enumerate(items, 1):
            lines.append(f"  {i}. \"{item}\"")
    return "\n".join(lines)


# ── LLM judge call ───────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, doubles each retry


async def call_judge(
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
) -> T:
    """Send a single LLM-as-Judge request and return a validated Pydantic model.

    Retries up to ``MAX_RETRIES`` times on transient errors with
    exponential backoff.  Uses sotopia's ``agenerate`` with
    ``PydanticOutputParser`` so the response conforms to
    *response_model*'s JSON schema.
    """
    from sotopia.generation_utils import agenerate
    from sotopia.generation_utils.generate import PydanticOutputParser

    full_prompt = system_prompt.rstrip() + "\n\n" + user_prompt.rstrip()

    # We use a single placeholder ``{prompt_text}`` to avoid issues with
    # curly braces that may appear inside transcript content.
    template = "{prompt_text}\n\n{format_instructions}"

    # When reasoning_effort is active (not None and not "none"),
    # temperature is incompatible and must not be sent.
    use_temperature: float | None = temperature
    if reasoning_effort is not None and reasoning_effort != "none":
        use_temperature = None

    last_exc: Exception | None = None
    extra_kwargs: dict[str, Any] = {}
    if reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = reasoning_effort
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result: T = await agenerate(
                model_name=model,
                template=template,
                input_values={"prompt_text": full_prompt},
                output_parser=PydanticOutputParser[response_model](pydantic_object=response_model),
                temperature=use_temperature,
                structured_output=True,
                **extra_kwargs,
            )
            return result
        except Exception as exc:
            last_exc = exc
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            logger.warning(
                "Judge call attempt %d/%d failed (%s: %s). Retrying in %.1fs…",
                attempt, MAX_RETRIES, type(exc).__name__, exc, wait,
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(wait)

    raise RuntimeError(
        f"Judge call failed after {MAX_RETRIES} attempts"
    ) from last_exc
