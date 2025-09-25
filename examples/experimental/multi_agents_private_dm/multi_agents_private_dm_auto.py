"""
Demo of private dm among three agents — config-driven & JSON-logged.
"""

# Run this in terminal first (if using redis persistence):
#   redis-stack-server --dir ./redis-data

import os
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Tuple, Sequence, Dict, List, cast

import redis

from sotopia.database.persistent_profile import AgentProfile, EnvironmentProfile
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server
from sotopia.messages import Observation, AgentAction, Message
from sotopia.messages import AgentAction as _AgentAction
from sotopia.messages import Observation as _Observation
from sotopia.messages import SimpleMessage as _SimpleMessage

# --- Evaluation imports ---
from itertools import chain
from sotopia.envs.evaluators import (
    EpisodeLLMEvaluator,
    EvaluationForAgents,
    unweighted_aggregate_evaluate,
)
from sotopia.database import SotopiaDimensions  # type: ignore

DM_PROTOCOL = """
# Messaging Protocol (MANDATORY)
- Public message → set "to": null
- Private/DM to a specific person → set "to": ["<their exact full name>"]
- Use the exact full name as shown in Participants.
- Never *say* “I will DM…”. Just send the DM by setting the field.

# Examples
- DM from Alice Smith to Ben Lee:
  {"action_type":"speak","argument":"Ben, quick one on clustering hyperparams…","to":["Ben Lee"]}

- Public message:
  {"action_type":"speak","argument":"Team, I suggest we start with the objectives.","to": null}
"""

# --- Redis (optional, as in your original script) ---
client = redis.Redis(host="localhost", port=6379)
os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")


# ----------------------------
# Utility: output directory + logging (text log to console only; JSON is written by us below)
# ----------------------------
def _mk_outdir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(__file__).resolve().parent / "runs" / ts
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


# def _setup_logging() -> None:
#     root = logging.getLogger()
#     root.setLevel(logging.INFO)
#     for h in list(root.handlers):
#         root.removeHandler(h)
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#     ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
#     root.addHandler(ch)

# this version shows some more details and avoids duplicate prints of "Generated result ..." lines
# def _setup_logging() -> None:
#     root = logging.getLogger()
#     root.setLevel(logging.INFO)

#     # remove pre-existing handlers (avoids duplicates on reruns)
#     for h in list(root.handlers):
#         root.removeHandler(h)

#     fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")

#     # console only (no file here since you’re writing JSON artifacts)
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#     ch.setFormatter(fmt)
#     root.addHandler(ch)

#     # QUIET DOWN third-party libs
#     # LiteLLM internal logger sometimes named "LiteLLM" or "litellm"
#     logging.getLogger("LiteLLM").setLevel(logging.ERROR)
#     logging.getLogger("litellm").setLevel(logging.ERROR)

#     # OpenAI / HTTP layer
#     logging.getLogger("httpx").setLevel(logging.WARNING)
#     logging.getLogger("urllib3").setLevel(logging.WARNING)
#     logging.getLogger("openai").setLevel(logging.WARNING)

#     # Keep sotopia outputs you like
#     logging.getLogger("sotopia").setLevel(logging.INFO)
#     logging.getLogger("sotopia.generation").setLevel(logging.INFO)


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # remove pre-existing handlers (avoids duplicates on reruns)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # QUIET DOWN third-party libs
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("litellm").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    # Keep sotopia logs you like
    logging.getLogger("sotopia").setLevel(logging.INFO)
    logging.getLogger("sotopia.generation").setLevel(logging.INFO)

    # 🔑 prevent duplicate prints of "Generated result ..." lines
    logging.getLogger("sotopia.generation").propagate = False

    # If you prefer to hide them entirely, comment the line above and use:
    # logging.getLogger("sotopia.generation").setLevel(logging.WARNING)


# Config loading
def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# DB helpers
def _get_or_create_agent(p: dict[str, Any]) -> AgentProfile:
    """
    p example:
    {
      "first_name": "...",
      "last_name": "...",
      "profile": {...},
      "goal": "..."
    }
    """
    fn = p["first_name"]
    ln = p["last_name"]

    existing = list(
        AgentProfile.find(
            AgentProfile.first_name == fn, AgentProfile.last_name == ln
        ).all()
    )
    if existing:
        return cast(AgentProfile, existing[0])

    # create
    prof = p.get("profile", {})
    agent = AgentProfile(
        first_name=fn,
        last_name=ln,
        age=prof.get("age"),
        occupation=prof.get("occupation"),
        gender=prof.get("gender"),
        gender_pronoun=prof.get("gender_pronoun"),
        big_five=prof.get("big_five"),
        moral_values=prof.get("moral_values"),
        decision_making_style=prof.get("decision_making_style"),
        secret=prof.get("secret"),
    )
    agent.save()
    return agent


def _create_env_profile(
    scenario: str, agent_goals: Sequence[str]
) -> EnvironmentProfile:
    env_profile = EnvironmentProfile(scenario=scenario, agent_goals=list(agent_goals))
    env_profile.save()
    return env_profile


# Serialization
# def _serialize_message(triple: Tuple[str, str, Message]) -> dict[str, Any]:
#     sender, receiver, msg = triple
#     rec: dict[str, Any] = {
#         "sender": sender,
#         "receiver": receiver,
#         "message_type": type(msg).__name__,
#         "natural": msg.to_natural_language(),
#     }
#     if isinstance(msg, _AgentAction):
#         rec.update({
#             "action_type": msg.action_type,
#             "argument": msg.argument,
#             "to": msg.to,   # None => public; list[str] => private recipients
#         })
#     elif isinstance(msg, _Observation):
#         rec.update({
#             "turn_number": msg.turn_number,
#             "available_actions": msg.available_actions,
#             "last_turn": msg.last_turn,
#         })
#     elif isinstance(msg, _ScriptBackground):
#         rec.update({
#             "scenario": msg.scenario
#         })
#     elif isinstance(msg, _SimpleMessage):
#         # already covered by "natural"
#         pass
#     return rec


def _serialize_message(triple):
    # guard for shape issues
    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
        return {"_skipped": True, "reason": "bad_shape", "repr": repr(triple)}

    sender, receiver, msg = triple
    rec = {
        "sender": sender,
        "receiver": receiver,
        "message_type": type(msg).__name__,
        "natural": msg.to_natural_language(),
    }
    # structured enrichment
    try:
        from sotopia.messages import (
            AgentAction as _AgentAction,
            Observation as _Observation,
        )
        from sotopia.messages import (
            SimpleMessage as _SimpleMessage,
            ScriptBackground as _ScriptBackground,
        )

        if isinstance(msg, _AgentAction):
            rec.update(
                {"action_type": msg.action_type, "argument": msg.argument, "to": msg.to}
            )
        elif isinstance(msg, _Observation):
            rec.update(
                {
                    "turn_number": msg.turn_number,
                    "available_actions": msg.available_actions,
                    "last_turn": msg.last_turn,
                }
            )
        elif isinstance(msg, _ScriptBackground):
            rec.update({"scenario": msg.scenario})
        elif isinstance(msg, _SimpleMessage):
            pass
    except Exception as e:
        rec["_enrich_error"] = str(e)
    return rec


def _save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _save_jsonl(records: List[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --- Evaluation helpers -------------------------------------------------------
def _build_history_from_triples(flat_triples: List[Tuple[str, str, Message]]) -> str:
    """
    Create a readable conversation transcript for the evaluator.
    Only includes 'speak' actions (public or private).
    """
    lines: List[str] = []
    for sender, _, m in flat_triples:
        if isinstance(m, _AgentAction) and m.action_type == "speak":
            to_str = "ALL" if (m.to is None) else ",".join(m.to)
            lines.append(f"{sender} -> [{to_str}]: {m.argument}")
    return "\n".join(lines)


def _to_jsonable(obj: Any) -> Any:
    """
    Convert evaluator outputs into JSON-serializable python objects.
    Works with Pydantic v1/v2, dataclasses, mappings, iterables, primitives.
    Falls back to repr() if needed.
    """
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    # Dataclasses
    try:
        import dataclasses

        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
    except Exception:
        pass
    # Mappings
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    # Iterables (but not strings/bytes)
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    # Primitives
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Fallback
    return repr(obj)


# Main
async def main() -> None:
    _setup_logging()
    outdir = _mk_outdir()

    # 1 load config
    cfg_path = (
        Path(os.environ.get("SCENARIO_JSON", ""))
        if os.environ.get("SCENARIO_JSON")
        else Path(__file__).with_name("scenario_config.json")
    )
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg = _load_config(cfg_path)

    scenario: str = cfg["scenario"]
    agents_cfg: List[dict[str, Any]] = cfg["agents"]
    models: dict[str, Any] = cfg.get(
        "models", {"env": "gpt-4o", "agents": ["gpt-4o", "gpt-4o", "gpt-4o"]}
    )
    action_order: str = cfg.get("action_order", "round-robin")

    # 2 ensure agents exist or create in DB, collect goals
    agent_profiles: List[AgentProfile] = []
    agent_fullnames: List[str] = []
    goals: List[str] = []

    for a in agents_cfg:
        ap = _get_or_create_agent(a)
        agent_profiles.append(ap)
        agent_fullnames.append(f"{ap.first_name} {ap.last_name}")
        goals.append(a.get("goal", ""))
    ## append DM protocol to each goal
    goals = [g + "\n" + DM_PROTOCOL for g in goals]

    # 3 create environment profile
    env_prof = _create_env_profile(scenario, goals)

    # 4 create sampler for exactly these 3 agents + env
    sampler: UniformSampler[Observation, AgentAction] = UniformSampler[
        Observation, AgentAction
    ](
        env_candidates=[env_prof],
        agent_candidates=agent_profiles,
    )

    # 5 build model_dict from config
    #    env + agent1..agentN (ordered)
    model_dict: Dict[str, str] = {"env": models["env"]}
    for i, m in enumerate(models.get("agents", []), start=1):
        model_dict[f"agent{i}"] = m

    logging.info(
        f"Running episode with agents: {agent_fullnames} | models: {model_dict}"
    )

    # 6 run the async server (returns list of episodes; here it's 1)
    episodes = await run_async_server(
        model_dict=model_dict,
        sampler=sampler,
        action_order=cast(Any, action_order),
    )
    # normal to a flat list of (sender, receiver, Message)
    # run_async_server -> arun_one_episode returns FLAT list[tuple] in non-streaming mode
    first = episodes[0]
    if first and isinstance(first[0], list):  # just in case some paths return nested
        flat_triples: List[Tuple[str, str, Message]] = [
            t
            for turn in cast(List[List[Tuple[str, str, Message]]], first)
            for t in turn
        ]
    else:
        flat_triples = cast(List[Tuple[str, str, Message]], first)

    # --- evaluate the episode and save to evaluations.json
    # Build evaluator transcript (only 'speak' actions)
    history_text = _build_history_from_triples(flat_triples)

    #  evaluator model use env model by default
    eval_model = model_dict.get("env", "gpt-4o")

    evaluator: EpisodeLLMEvaluator[SotopiaDimensions] = EpisodeLLMEvaluator(
        model_name=eval_model,
        response_format_class=EvaluationForAgents[SotopiaDimensions],
    )

    eval_raw_list = await asyncio.gather(
        evaluator.__acall__(
            turn_number=-1,
            history=history_text,
            messages=None,
            temperature=0.0,
        )
    )
    eval_agg = unweighted_aggregate_evaluate(list(chain(*eval_raw_list)))

    evaluations_payload = {
        "model": eval_model,
        "agents_order": agent_fullnames,
        "history_char_count": len(history_text),
        # include history if  want the full text the evaluator saw:
        # "history": history_text,
        "result": _to_jsonable(eval_agg),
    }
    _save_json(evaluations_payload, outdir / "evaluations.json")

    # 7 outputs (JSON)
    # 7a  run report (scenario, agents, models, counts)
    turn_markers = sum(
        1
        for (_, _, m) in flat_triples
        if isinstance(m, _SimpleMessage)
        and m.to_natural_language().startswith("Turn #")
    )
    max_obs_turn = 0
    for _, _, m in flat_triples:
        if isinstance(m, _Observation):
            if m.turn_number is not None:
                max_obs_turn = max(max_obs_turn, m.turn_number)

    report = {
        "scenario": scenario,
        "agents": [
            {
                "name": agent_fullnames[i],
                "profile": {
                    "age": agent_profiles[i].age,
                    "occupation": agent_profiles[i].occupation,
                    "gender": agent_profiles[i].gender,
                    "gender_pronoun": agent_profiles[i].gender_pronoun,
                    "big_five": agent_profiles[i].big_five,
                    "moral_values": agent_profiles[i].moral_values,
                    "decision_making_style": agent_profiles[i].decision_making_style,
                    "secret": agent_profiles[i].secret,
                },
                "goal": goals[i],
            }
            for i in range(len(agent_profiles))
        ],
        "models": model_dict,
        "action_order": action_order,
        "stats": {
            "turns_from_markers": turn_markers,
            "max_observation_turn": max_obs_turn,
        },
    }
    _save_json(report, outdir / "run_report.json")

    # 7b full per-message log as JSONL
    # flat_records: List[dict[str, Any]] = []
    # for turn in nested_messages:
    #     for triple in turn:
    #         flat_records.append(_serialize_message(triple))
    # flat_records = []
    # for turn in nested_messages:
    #     if not isinstance(turn, (list, tuple)):
    #         flat_records.append({"_skipped_turn": True, "repr": repr(turn)})
    #         continue
    #     for item in turn:
    #         flat_records.append(_serialize_message(item))

    # _save_jsonl(flat_records, outdir / "messages.jsonl")
    records = [_serialize_message(triple) for triple in flat_triples]
    _save_jsonl(records, outdir / "messages.jsonl")

    logging.info(f"Saved JSON outputs → {outdir}")


if __name__ == "__main__":
    asyncio.run(main())
