"""
Run structured multi-agent scenarios from JSON and save transcripts.

Input JSON may be one of:
- A list of scenario objects
- A dict with key "scenarios" -> list
- A single scenario object

Notes
- This script focuses on running simulations and writing transcripts and per-agent views.
- Metrics are optional; if you pass --metrics and the metric modules are importable, they will run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, TypedDict, Optional, Literal, cast

# Ensure the benchmarks directory is on sys.path so metric modules are importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from sotopia.database.persistent_profile import AgentProfile, EnvironmentProfile
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    RuleBasedTerminatedEvaluator,
    EpisodeLLMEvaluator,
    EvaluationForAgents,
)
from sotopia.database import SotopiaDimensions
from sotopia.messages import AgentAction, SimpleMessage, Observation
from sotopia.agents.strategy_llm_agent import StrategyLLMAgent, PromptMode


class KnowledgeItem(TypedDict, total=False):
    value: Any
    sensitivity: str


class AgentSpec(TypedDict):
    agent_id: int
    role: str
    pre_interaction_knowledge: dict[str, KnowledgeItem]
    post_interaction_knowledge: dict[str, Any]
    goals: dict[str, Any]


class ScenarioSpec(TypedDict):
    scenario_id: int
    sector: str
    scenario_goal: str
    knowledge_domain_map: dict[str, Any]
    agents: list[AgentSpec]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_goal_text(agent: AgentSpec) -> str:
    role = agent["role"]
    primary = agent["goals"].get("primary_objective", "")
    sharing = agent["goals"].get("sharing_policy", {})
    pre_json = json.dumps(
        agent["pre_interaction_knowledge"], indent=2, ensure_ascii=False
    )
    share_list = sharing.get("what_to_share", [])
    not_share_list = sharing.get("what_not_to_share", [])

    lines: list[str] = []
    lines.append(f"ROLE: {role}")
    lines.append("")
    lines.append("PRIMARY OBJECTIVE:")
    lines.append(primary if primary else "(none provided)")
    lines.append("")
    lines.append("PRE-INTERACTION KNOWLEDGE YOU CURRENTLY HOLD:")
    lines.append(pre_json)
    lines.append("")
    lines.append("SHARING POLICY:")
    lines.append("- You MAY share:")
    lines.extend([f"  • {item}" for item in share_list] or ["  • (none specified)"])
    lines.append("- You MUST NOT share:")
    lines.extend([f"  • {item}" for item in not_share_list] or ["  • (none specified)"])
    lines.append("")
    lines.append("REMINDER:")
    lines.append(
        "- Use PRIVATE channel ('to' field) for sensitive topics; public if 'to' is empty."
    )
    lines.append("- Respect policy: never disclose items in 'MUST NOT share'.")

    return "\n".join(lines)


def to_agent_profile(agent: AgentSpec, tag: str) -> AgentProfile:
    # Split pre-knowledge by sensitivity into public vs private summary strings
    public_parts: list[str] = []
    private_parts: list[str] = []
    for k, item in agent["pre_interaction_knowledge"].items():
        sens = item.get("sensitivity", "public")
        val = item.get("value")
        if sens == "private":
            private_parts.append(f"{k}: {val}")
        else:
            public_parts.append(f"{k}: {val}")

    profile = AgentProfile(
        first_name=agent["role"],
        last_name="",
        occupation=agent["role"],
        public_info="; ".join(public_parts),
        secret="; ".join(private_parts),
        tag=tag,
    )
    profile.save()
    return profile


def to_environment_profile(
    spec: ScenarioSpec, agent_goals: list[str], tag: str
) -> EnvironmentProfile:
    scenario_text = (
        f"Scenario Objective: {spec['scenario_goal']}\n"
        "Knowledge domain ownership (JSON):\n"
        + json.dumps(spec["knowledge_domain_map"], indent=2, ensure_ascii=False)
    )

    env = EnvironmentProfile(
        codename=f"scenario_{spec['scenario_id']}",
        scenario=scenario_text,
        agent_goals=agent_goals,
        tag=tag,
    )
    env.save()
    return env


def build_env_and_agents(
    spec: ScenarioSpec,
    *,
    agent_model: str,
    env_model: str,
    action_order: Literal["simultaneous", "round-robin", "random"] = "round-robin",
    prompt_mode: PromptMode = "basic",
    disable_terminal_eval: bool = False,
    max_turns: int = 20,
):
    tag = f"scenario_{spec['scenario_id']}"

    agent_profiles: list[AgentProfile] = []
    agent_goals: list[str] = []
    for agent in spec["agents"]:
        agent_profiles.append(to_agent_profile(agent, tag=tag))
        agent_goals.append(build_goal_text(agent))

    env = to_environment_profile(spec, agent_goals, tag=tag)

    # Auto-disable terminal eval for non-OpenAI env models (structured output not supported)
    _env_is_openai = not any(
        env_model.startswith(p)
        for p in ("together_ai/", "anthropic/", "huggingface/", "ollama/", "replicate/")
    )
    use_terminal_eval = (not disable_terminal_eval) and _env_is_openai

    sim_env = ParallelSotopiaEnv(
        model_name=env_model,
        action_order=action_order,
        evaluators=[
            RuleBasedTerminatedEvaluator(max_turn_number=max_turns, max_stale_turn=4)
        ],
        terminal_evaluators=(
            [EpisodeLLMEvaluator(env_model, EvaluationForAgents[SotopiaDimensions])]
            if use_terminal_eval
            else []
        ),
        env_profile=env,
    )

    agents_list = [
        StrategyLLMAgent(
            agent_profile=ap, model_name=agent_model, prompt_mode=prompt_mode
        )
        for ap in agent_profiles
    ]
    return sim_env, agents_list


def flatten_episode(episode: list[Any]) -> list[tuple[str, str, object]]:
    flat: list[tuple[str, str, object]] = []
    for item in episode:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 3
            and isinstance(item[0], str)
        ):
            flat.append(item)  # type: ignore[arg-type]
        elif isinstance(item, list):
            for sub in item:
                if (
                    isinstance(sub, (list, tuple))
                    and len(sub) == 3
                    and isinstance(sub[0], str)
                ):
                    flat.append(sub)  # type: ignore[arg-type]
    return flat


def write_scenario_outputs(
    *,
    spec: ScenarioSpec,
    flat: list[tuple[str, str, object]],
    out_dir: str,
) -> None:
    scenario_dir = os.path.join(out_dir, str(spec["scenario_id"]))
    _ensure_dir(scenario_dir)

    # Save the scenario spec so downstream tools (aggregator) can read it
    with open(os.path.join(scenario_dir, "spec.json"), "w") as sf:
        json.dump(dict(spec), sf, indent=2)  # type: ignore[arg-type]

    txt_path = os.path.join(scenario_dir, "transcript.txt")
    jsonl_path = os.path.join(scenario_dir, "transcript.jsonl")

    with open(txt_path, "w") as f_txt, open(jsonl_path, "w") as f_jsonl:
        for sender, receiver, msg in flat:
            if hasattr(msg, "to_natural_language"):
                line = f"{sender} -> {receiver}: {msg.to_natural_language()}\n"  # type: ignore[attr-defined]
            else:
                line = f"{sender} -> {receiver}: {str(msg)}\n"
            f_txt.write(line)

            entry: dict[str, Any] = {"sender": sender, "receiver": receiver}
            if isinstance(msg, AgentAction):
                entry.update(
                    {
                        "type": "agent_action",
                        "action_type": msg.action_type,
                        "argument": msg.argument,
                        "to": msg.to,
                    }
                )
            elif isinstance(msg, SimpleMessage):
                entry.update({"type": "message", "text": msg.to_natural_language()})
            else:
                entry.update({"type": "unknown", "repr": str(msg)})
            f_jsonl.write(json.dumps(entry) + "\n")

    # ── Build structured round data ────────────────────────────────────
    # In round-robin with N agents, one "round" = N consecutive sim-turns.

    # Step 1: discover agent names in speaking order
    agent_names_ordered: list[str] = []
    for sender, receiver, msg in flat:
        if (
            receiver == "Environment"
            and sender != "Environment"
            and isinstance(msg, AgentAction)
            and sender not in agent_names_ordered
        ):
            agent_names_ordered.append(sender)
    num_agents = max(len(agent_names_ordered), 1)

    # Build short aliases for compact display  (e.g. "SRE Lead" for long role names)
    agent_aliases: dict[str, str] = {}
    for name in agent_names_ordered:
        # Use the name as-is; callers can shorten in JSON if desired
        agent_aliases[name] = name

    # Step 2: collect per-sim-turn structured utterances
    class Utterance:
        __slots__ = ("sender", "action_type", "to", "argument")

        def __init__(
            self, sender: str, action_type: str, to: list[str] | None, argument: str
        ):
            self.sender = sender
            self.action_type = action_type
            self.to = to or []
            self.argument = argument

        @property
        def visibility(self) -> str:
            return f"private to={','.join(self.to)}" if self.to else "public"

    sim_turn_utterances: list[list[Utterance]] = []
    current_utts: list[Utterance] = []
    last_seen_turn = -1
    for sender, receiver, msg in flat:
        if sender == "Environment" and isinstance(msg, Observation):
            tn = getattr(msg, "turn_number", -1)
            if isinstance(tn, int) and tn > last_seen_turn:
                if current_utts:
                    sim_turn_utterances.append(current_utts)
                    current_utts = []
                last_seen_turn = tn
            continue
        if (
            receiver == "Environment"
            and sender != "Environment"
            and isinstance(msg, AgentAction)
        ):
            if msg.action_type == "none":
                continue
            current_utts.append(
                Utterance(sender, msg.action_type, msg.to, msg.argument)
            )
    if current_utts:
        sim_turn_utterances.append(current_utts)

    # Step 3: merge every `num_agents` sim-turns into one round
    rounds: list[list[Utterance]] = []
    for i in range(0, len(sim_turn_utterances), num_agents):
        round_utts: list[Utterance] = []
        for block in sim_turn_utterances[i : i + num_agents]:
            round_utts.extend(block)
        if round_utts:
            rounds.append(round_utts)

    # ── transcript_pretty.txt  (judge-friendly, citable IDs) ─────────
    pretty_path = os.path.join(scenario_dir, "transcript_pretty.txt")
    with open(pretty_path, "w") as f:
        # Header block
        agent_list_str = ", ".join(
            f"{idx + 1}={name}" for idx, name in enumerate(agent_names_ordered)
        )
        f.write(
            f"[SCENARIO] id={spec['scenario_id']} | sector={spec.get('sector', '?')} | agents={num_agents} | rounds={len(rounds)}\n"
        )
        f.write(f"[AGENTS] {agent_list_str}\n")
        f.write(f"[GOAL] {spec['scenario_goal']}\n")
        f.write("\n")

        if not rounds:
            f.write("No rounds detected. Raw lines were written to transcript.txt.\n")
        else:
            for r_idx, round_utts in enumerate(rounds):
                for u_idx, utt in enumerate(round_utts):
                    tag = f"[R{r_idx}.{u_idx + 1}]"
                    f.write(
                        f"{tag} {utt.sender} ({utt.action_type}, {utt.visibility}): {utt.argument}\n"
                    )
                f.write("\n")

    # ── Per-agent filtered views (same citable IDs) ──────────────────
    views_dir = os.path.join(scenario_dir, "views")
    _ensure_dir(views_dir)
    for viewer in agent_names_ordered:
        safe_name = viewer.replace(" ", "_").replace("/", "_").lower()
        view_path = os.path.join(views_dir, f"{safe_name}_view.txt")
        with open(view_path, "w") as vf:
            vf.write(
                f"[VIEW] agent={viewer} | scenario_id={spec['scenario_id']} | rounds={len(rounds)}\n\n"
            )
            for r_idx, round_utts in enumerate(rounds):
                header_written = False
                for u_idx, utt in enumerate(round_utts):
                    visible = (
                        not utt.to  # public
                        or utt.sender == viewer  # viewer sent it
                        or viewer in utt.to  # viewer is a recipient
                    )
                    if visible:
                        if not header_written:
                            vf.write(f"--- Round {r_idx} ---\n")
                            header_written = True
                        tag = f"[R{r_idx}.{u_idx + 1}]"
                        vf.write(
                            f"{tag} {utt.sender} ({utt.action_type}, {utt.visibility}): {utt.argument}\n"
                        )
                if header_written:
                    vf.write("\n")


async def amain(args: argparse.Namespace) -> None:
    # Load scenarios JSON
    if not args.json:
        raise SystemExit("--json path is required")
    with open(args.json, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "scenarios" in data:
        scenarios: list[ScenarioSpec] = data["scenarios"]  # type: ignore[assignment]
    elif isinstance(data, list):
        scenarios = data  # type: ignore[assignment]
    else:
        scenarios = [data]  # type: ignore[list-item]

    # Optional filter
    if args.scenario_id is not None:
        sid = int(args.scenario_id)
        scenarios = [s for s in scenarios if int(s.get("scenario_id", -1)) == sid]
        if not scenarios:
            raise SystemExit(
                f"No scenario with scenario_id={sid} found in {args.json}."
            )

    if args.max_scenarios is not None:
        scenarios = scenarios[: args.max_scenarios]
        print(
            f"Limiting to first {args.max_scenarios} scenarios ({len(scenarios)} loaded)."
        )

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.json))
    _ensure_dir(out_dir)
    eval_root = os.path.join(out_dir, "scenario_eval")
    _ensure_dir(eval_root)
    if not os.getenv("SOTOPIA_TOM_DIR"):
        os.environ["SOTOPIA_TOM_DIR"] = eval_root

    # Resolve models
    agent_model = args.agent_model or os.environ.get("AGENT_MODEL") or "gpt-4o-mini"
    env_model = args.env_model or os.environ.get("ENV_MODEL") or "gpt-4o"

    # Build env/agent combos
    combos = []
    for spec in scenarios:
        combos.append(
            build_env_and_agents(
                spec,
                agent_model=agent_model,
                env_model=env_model,
                action_order=args.action_order,  # type: ignore[arg-type]
                prompt_mode=cast(PromptMode, args.prompt_mode),
                disable_terminal_eval=args.disable_terminal_eval,
                max_turns=args.max_turns,
            )
        )

    # Run simulations in batches
    from sotopia.server import run_async_server

    batch_size = args.batch_size
    all_results: list[tuple] = []  # (spec, episode) pairs

    for batch_start in range(0, len(scenarios), batch_size):
        batch_specs = scenarios[batch_start : batch_start + batch_size]
        batch_combos = combos[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(scenarios) + batch_size - 1) // batch_size
        print(f"\n{'='*60}")
        print(
            f"Running batch {batch_num}/{total_batches} (scenarios {batch_start+1}-{batch_start+len(batch_specs)} of {len(scenarios)})"
        )
        print(f"{'='*60}")

        batch_results = await run_async_server(
            env_agent_combo_list=batch_combos,
            action_order=args.action_order,  # type: ignore[arg-type]
        )

        for spec, episode in zip(batch_specs, batch_results):
            all_results.append((spec, episode))

        # Small delay between batches to help with rate limits
        if batch_start + batch_size < len(scenarios):
            print(f"Batch {batch_num} complete. Pausing 5s before next batch...")
            await asyncio.sleep(5)

    # Save outputs and optional metrics
    for spec, episode in all_results:
        flat = flatten_episode(episode)
        write_scenario_outputs(
            spec=spec,
            flat=flat,
            out_dir=eval_root,
        )

        if args.metrics:
            scenario_dir = os.path.join(eval_root, str(spec["scenario_id"]))
            errors: list[str] = []
            try:
                from metrics_da import compute_and_save_da  # type: ignore
                from metrics_ia import compute_and_save_ia  # type: ignore
                from metrics_eff import compute_and_save_eff  # type: ignore
                from metrics_cpv import compute_and_save_cpv  # type: ignore
                from metrics_composite import compute_and_save_composite  # type: ignore
                from aggregate_new_llm import aggregate_new_llm_metrics  # type: ignore

                judge = args.judge_model or env_model
                re = args.judge_reasoning_effort
                results = await asyncio.gather(
                    compute_and_save_da(
                        spec=spec,
                        flat_messages=flat,
                        scenario_dir=scenario_dir,
                        judge_model=judge,
                        reasoning_effort=re,
                    ),
                    compute_and_save_ia(
                        spec=spec,
                        flat_messages=flat,
                        scenario_dir=scenario_dir,
                        judge_model=judge,
                        reasoning_effort=re,
                    ),
                    compute_and_save_eff(
                        spec=spec,
                        flat_messages=flat,
                        scenario_dir=scenario_dir,
                        judge_model=judge,
                        reasoning_effort=re,
                    ),
                    compute_and_save_cpv(
                        spec=spec,
                        flat_messages=flat,
                        scenario_dir=scenario_dir,
                        judge_model=judge,
                        reasoning_effort=re,
                    ),
                    return_exceptions=True,
                )
                for r in results:
                    if isinstance(r, Exception):
                        errors.append(repr(r))
                compute_and_save_composite(scenario_dir=scenario_dir)
                aggregate_new_llm_metrics(eval_root)
            except Exception as e:  # imports or runtime errors
                errors.append(repr(e))
            if errors:
                mdir = os.path.join(scenario_dir, "metrics")
                _ensure_dir(mdir)
                with open(os.path.join(mdir, "errors.txt"), "a") as ef:
                    ef.write("\n".join(errors) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run structured scenarios from JSON (transcripts + optional metrics)"
    )
    p.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to JSON file containing scenarios or a single scenario",
    )
    p.add_argument(
        "--agent-model",
        type=str,
        default=None,
        help="LLM to use for all agents (ENV: AGENT_MODEL)",
    )
    p.add_argument(
        "--env-model",
        type=str,
        default=None,
        help="LLM to use for environment/evaluator (ENV: ENV_MODEL)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write outputs; default is the JSON's directory",
    )
    p.add_argument(
        "--action-order",
        type=str,
        choices=["simultaneous", "round-robin", "random"],
        default="round-robin",
        help="Agent action scheduling policy",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="LLM to use as the judge for terminal evaluation (defaults to --env-model if not set)",
    )
    p.add_argument(
        "--disable-terminal-eval",
        action="store_true",
        help="Disable SotopiaDimensions terminal evaluation at the end of each simulation",
    )
    p.add_argument(
        "--scenario-id",
        type=int,
        default=None,
        help="If provided, run only the scenario with this id",
    )
    p.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="If provided, run only the first N scenarios from the dataset",
    )
    p.add_argument(
        "--prompt-mode",
        type=str,
        choices=["basic", "cot", "tom", "tom_coach", "tom_belief", "tom_scratchpad"],
        default="basic",
        help="Prompting strategy: basic / CoT / ToM (prompt-only) / ToM-Coach (stateless LLM advisor) / ToM-Belief (stateful belief tracker) / ToM-Scratchpad (ablation: generic scratchpad replacing belief slots)",
    )
    p.add_argument(
        "--metrics",
        action="store_true",
        help="Attempt to run external metric modules if available",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of scenarios to run per batch (default 5). Lower to avoid rate limits.",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=60,
        help="Maximum number of turns per simulation (default 60)",
    )
    p.add_argument(
        "--judge-reasoning-effort",
        type=str,
        choices=["none", "low", "medium", "high", "xhigh"],
        default=None,
        help="Reasoning effort for judge model (none/low/medium/high/xhigh). xhigh only for gpt-5.2-pro.",
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
