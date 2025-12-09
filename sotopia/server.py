import asyncio
import json
import itertools
import logging
import re
from typing import Literal, Sequence, Type, AsyncGenerator, Union, Any

import gin
from pydantic import validate_call
import rich

from sotopia.agents import (
    Agents,
    HumanAgent,
    LLMAgent,
    RedisAgent,
    ScriptWritingAgent,
)
from sotopia.agents.base_agent import BaseAgent
from sotopia.database import EpisodeLog, NonStreamingSimulationStatus, SotopiaDimensions
from sotopia.envs import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    EvaluationForAgents,
    EpisodeLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    unweighted_aggregate_evaluate,
)
from sotopia.generation_utils.generate import agenerate_script
from sotopia.messages import AgentAction, Message, Observation, SimpleMessage
from sotopia.messages.message_classes import (
    ScriptBackground,
    ScriptEnvironmentResponse,
)
from sotopia.samplers import BaseSampler, EnvAgentCombo


@validate_call
def run_sync_server(
    model_name_dict: dict[str, str],
    action_order: Literal["simultaneous", "round-robin", "random"],
    agents_info: dict[str, dict[str, str]] | None = None,
    partial_background_file: str | None = None,
    full_background_file: str | None = None,
    mode: str | None = None,
) -> list[tuple[str, str, Message]]:
    # Create Environment and agents
    # This step will be moved to outside this function

    env = ParallelSotopiaEnv(
        action_order=action_order,
        model_name=model_name_dict["env"],
        evaluators=[
            RuleBasedTerminatedEvaluator(),
        ],
    )
    if partial_background_file:
        environment_messages = env.reset(
            options={"partial_background_file": partial_background_file}
        )
    elif full_background_file:
        environment_messages = env.reset(
            options={"full_background_file": full_background_file}
        )
    else:
        environment_messages = env.reset()
    agents = Agents()
    # derive agent keys like agent1, agent2, … agentN
    agent_keys = sorted(k for k in model_name_dict if re.fullmatch(r"agent\d+", k))
    agents_model_names = [model_name_dict[k] for k in agent_keys]
    if len(agents_model_names) != len(env.agents):
        raise ValueError("Number of agent models must match number of env agents")

    for agent_name, agent_model in zip(env.agents, agents_model_names):
        if agent_model == "human":
            agents[agent_name] = HumanAgent(agent_name)
        elif mode == "speak":
            raise NotImplementedError(
                "Deprecated. The original Speaker Agent is not implemented in the async context."
            )
        else:
            agents[agent_name] = LLMAgent(agent_name, model_name=agent_model)
    agents.reset()

    messages: list[tuple[str, str, Message]] = []

    # Main Event Loop
    done = False
    for agent_name in env.agents:
        messages.append(("Environment", agent_name, environment_messages[agent_name]))

    while not done:
        # gather agent messages
        agent_messages: dict[str, AgentAction] = dict()
        for agent_name in env.agents:
            if agents_info is not None:
                agents[agent_name].goal = agents_info[agent_name]["goal"]
            agent_messages[agent_name] = agents[agent_name].act(
                environment_messages[agent_name]
            )
            messages.append((agent_name, "Environment", agent_messages[agent_name]))

        # send agent messages to environment
        environment_messages, _, terminated, ___, ____ = env.step(agent_messages)
        for agent_name in env.agents:
            messages.append(
                ("Environment", agent_name, environment_messages[agent_name])
            )
        done = all(terminated.values())

    return messages


def flatten_listed_messages(
    messages: list[list[tuple[str, str, Message]]],
) -> list[tuple[str, str, Message]]:
    return list(itertools.chain.from_iterable(messages))


@gin.configurable
async def arun_one_episode(
    env: ParallelSotopiaEnv,
    agent_list: Sequence[BaseAgent[Observation, AgentAction]],
    omniscient: bool = False,
    script_like: bool = False,
    json_in_script: bool = False,
    tag: str | None = None,
    push_to_db: bool = False,
    episode_pk: str | None = None,
    streaming: bool = False,
    simulation_status: NonStreamingSimulationStatus | None = None,
    output_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Union[
    list[tuple[str, str, Message]],
    AsyncGenerator[list[list[tuple[str, str, Message]]], None],
]:
    agents = Agents({agent.agent_name: agent for agent in agent_list})

    async def generate_messages() -> (
        AsyncGenerator[list[list[tuple[str, str, Message]]], None]
    ):
        environment_messages = env.reset(agents=agents, omniscient=omniscient)
        agents.reset()
        messages: list[list[tuple[str, str, Message]]] = []

        # Main Event Loop
        done = False
        messages.append(
            [
                ("Environment", agent_name, environment_messages[agent_name])
                for agent_name in env.agents
            ]
        )
        yield messages

        # set goal for agents
        for index, agent_name in enumerate(env.agents):
            agents[agent_name].goal = env.profile.agent_goals[index]
        rewards: list[list[float]] = []
        reasons: list[str] = []
        while not done:
            # gather agent messages
            agent_messages: dict[str, AgentAction] = dict()

            actions = await asyncio.gather(
                *[
                    agents[agent_name].aact(environment_messages[agent_name])
                    for agent_name in env.agents
                ]
            )
            if script_like:
                # manually mask one message
                agent_mask = env.action_mask
                for idx in range(len(agent_mask)):
                    if agent_mask[idx] == 0:
                        actions[idx] = AgentAction(action_type="none", argument="")
                    else:
                        pass

            for idx, agent_name in enumerate(env.agents):
                agent_messages[agent_name] = actions[idx]

                messages[-1].append(
                    (agent_name, "Environment", agent_messages[agent_name])
                )

            # send agent messages to environment
            (
                environment_messages,
                rewards_in_turn,
                terminated,
                ___,
                info,
            ) = await env.astep(agent_messages)
            messages.append(
                [
                    ("Environment", agent_name, environment_messages[agent_name])
                    for agent_name in env.agents
                ]
            )
            yield messages
            rewards.append([rewards_in_turn[agent_name] for agent_name in env.agents])
            reasons.append(
                " ".join(info[agent_name]["comments"] for agent_name in env.agents)
            )
            done = all(terminated.values())

        epilog = EpisodeLog(
            environment=env.profile.pk,
            agents=[agent.profile.pk for agent in agent_list],
            tag=tag,
            models=[env.model_name] + [agent.model_name for agent in agent_list],
            messages=[
                [(m[0], m[1], m[2].to_natural_language()) for m in messages_in_turn]
                for messages_in_turn in messages
            ],
            reasoning=info[env.agents[0]]["comments"],
            rewards=[info[agent_name]["complete_rating"] for agent_name in env.agents],
            metadata=metadata or {},
        )

        if streaming:
            # yield the rewards and reasonings
            messages.append(
                [("Evaluation", "Rewards", SimpleMessage(message=str(epilog.rewards)))]
            )
            messages.append(
                [("Evaluation", "Reasoning", SimpleMessage(message=epilog.reasoning))]
            )
            yield messages

        if push_to_db:
            try:
                if episode_pk:
                    epilog.pk = episode_pk
                    epilog.save()
                else:
                    epilog.save()
                if simulation_status:
                    simulation_status.status = "Completed"
                    simulation_status.save()
            except Exception as e:
                logging.error(f"Failed to save episode log: {e}")

        if output_path:
            try:
                # Construct simplified log
                model_mapping = {}
                for agent_name, agent in agents.items():
                    model_mapping[agent_name] = agent.model_name

                turns_list = []
                for agent_name, agent in agents.items():
                    if hasattr(agent, "generation_history"):
                        for entry in agent.generation_history:
                            # Flatten and clean up entry
                            clean_entry = {
                                "turn_number": entry.get("turn_number"),
                                "agent_name": agent_name,
                                "model_name": model_mapping.get(agent_name, "Unknown"),
                                "prompt": entry.get("prompt"),
                                "response": entry.get("response"),
                            }
                            turns_list.append(clean_entry)

                # Sort by turn number
                turns_list.sort(
                    key=lambda x: int(x["turn_number"])
                    if x["turn_number"] is not None
                    else -1
                )

                custom_log = {
                    "pk": epilog.pk,
                    "tag": tag,
                    "metadata": metadata or {},
                    "model_mapping": model_mapping,
                    "rewards": epilog.rewards,
                    "turns": turns_list,
                }

                with open(output_path, "w") as f:
                    # Use json.dumps with indent for readability
                    f.write(json.dumps(custom_log, indent=4))
            except Exception as e:
                import traceback

                logging.error(
                    f"Failed to save episode log to file: {e}\n{traceback.format_exc()}"
                )

    if streaming:
        return generate_messages()
    else:
        async for last_messages in generate_messages():
            pass
        return flatten_listed_messages(last_messages)


@gin.configurable
async def run_async_server(
    sampler: BaseSampler[Observation, AgentAction] = BaseSampler(),
    action_order: Literal["simultaneous", "round-robin", "random"] = "round-robin",
    model_dict: dict[str, str] = {},
    env_agent_combo_list: list[EnvAgentCombo[Observation, AgentAction]] = [],
    omniscient: bool = False,
    script_like: bool = False,
    json_in_script: bool = False,
    tag: str | None = None,
    push_to_db: bool = False,
    using_async: bool = True,
) -> list[list[tuple[str, str, Message]]]:
    """
    Doc incomplete

    Args:
        omniscient (bool): Whether the agent knows the goal of the other, default to False
        script_like (bool): Whether we generate the turn in script like manner, default to False
        json_in_script (bool): Whether we requires the script generator to return json (Only valid when script_like is True), default to False

    Note: env_agent_combo_list is optional. When it defaults to [], sampler is used
    else the sampler is not used. Please pass in BaseSampler or simply not specify it when using this option.
    """

    assert not (push_to_db and tag is None), "please provide a tag when push to db"
    assert (
        model_dict or env_agent_combo_list
    ), "please provide model_dict or env_agent_combo_list"

    # Create Environment and agents
    # This step will be moved to outside this function

    def get_agent_class(
        model_name: str,
    ) -> Type[BaseAgent[Observation, AgentAction]]:
        if model_name == "human":
            return HumanAgent
        elif model_name == "redis":
            return RedisAgent
        elif script_like and not json_in_script:
            return ScriptWritingAgent
        else:
            return LLMAgent

    if env_agent_combo_list:
        assert (
            type(sampler) is BaseSampler
        ), "No sampler should be used when `env_agent_combo_list` is not empty"
        env_agent_combo_iter = iter(env_agent_combo_list)
    else:
        env_params = {
            "model_name": model_dict["env"],
            "action_order": action_order,
            "evaluators": [
                RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
            ],
            "terminal_evaluators": [
                EpisodeLLMEvaluator(
                    model_dict.get("evaluator", model_dict["env"]),
                    EvaluationForAgents[SotopiaDimensions],
                ),
            ],
        }

        agent_keys = sorted(k for k in model_dict if re.fullmatch(r"agent\d+", k))
        agent_models = [model_dict[k] for k in agent_keys]

        agents_model_dict = {
            f"agent{i+1}": model_name for i, model_name in enumerate(agent_models)
        }

        env_agent_combo_iter = sampler.sample(
            agent_classes=[
                get_agent_class(model_name) for model_name in agents_model_dict.values()
            ],
            n_agent=len(agents_model_dict),
            env_params=env_params,
            agents_params=[
                {"model_name": model_name} if model_name != "human" else {}
                for model_name in agents_model_dict.values()
            ],
        )
    episode_futures = [
        arun_one_episode(
            env=env_agent_combo[0],
            agent_list=env_agent_combo[1],
            omniscient=omniscient,
            script_like=script_like,
            json_in_script=json_in_script,
            tag=tag,
            push_to_db=push_to_db,
        )
        for env_agent_combo in env_agent_combo_iter
    ]

    batch_results = (
        await asyncio.gather(*episode_futures)
        if using_async
        else [await i for i in episode_futures]
    )

    if len(batch_results) > 0:
        first_result = batch_results[0]
        assert isinstance(
            first_result, list
        ), f"Unexpected result type: {type(first_result)}"

    return batch_results  # type: ignore


async def arun_one_script(
    env: ParallelSotopiaEnv,
    agent_list: Sequence[BaseAgent[Observation, AgentAction]],
    model_dict: dict[str, str],
    omniscient: bool = False,
    tag: str | None = None,
    push_to_db: bool = False,
) -> list[tuple[str, str, Message]]:
    """
    Generate script for one episode
    Args:
        omniscient (bool): Whether the agent knows the goal of the other
    """

    agents = Agents({agent.agent_name: agent for agent in agent_list})
    env.reset(agents=agents, omniscient=omniscient)

    agent_names = [agent.agent_name for agent in agent_list]
    # assert len(agent_names) == 2, f"only support 2 agents, current: {agent_names}"
    assert (
        agents and len(agents) >= 2
    ), "At least two agents required, current: {agent_names}"

    script_background = env.inbox[0][1]
    assert isinstance(script_background, ScriptBackground)
    story, prompt = await agenerate_script(
        model_name=model_dict["env"],
        background=script_background,
        agent_names=agent_names,
    )
    messages, agent_messages = story
    env_message = [("Environment", script_background)]
    agent_messages = env_message + agent_messages

    evaluator: EpisodeLLMEvaluator[SotopiaDimensions] = EpisodeLLMEvaluator(
        model_name=model_dict.get("evaluator", model_dict["env"]),
        response_format_class=EvaluationForAgents[SotopiaDimensions],
    )
    response = unweighted_aggregate_evaluate(
        list(
            itertools.chain(
                *await asyncio.gather(
                    *[
                        sing_evaluator.__acall__(
                            turn_number=-1,
                            messages=agent_messages,
                        )
                        for sing_evaluator in [evaluator]
                    ]
                )
            )
        )
    )
    info: dict[str, dict[str, str | ScriptEnvironmentResponse | float | None]] = {
        script_background.agent_names[0]: {
            "comments": response.comments or "",
            "complete_rating": response.p1_rate or 0,  # type: ignore
        },
        script_background.agent_names[1]: {
            "comments": response.comments or "",
            "complete_rating": response.p2_rate or 0,  # type: ignore
        },
        "rewards_prompt": {"overall_prompt": evaluator.prompt or ""},
    }
    epilog = EpisodeLog(
        environment=env.profile.pk,
        agents=[agent.profile.pk for agent in agent_list],
        tag=tag,
        models=[model_dict["env"]]
        + [
            model_dict.get(f"agent{i+1}", model_dict.get("agent1", ""))
            for i in range(len(agent_list))
        ],
        messages=[
            [(m[0], m[1], m[2].to_natural_language()) for m in messages_in_turn]
            for messages_in_turn in messages
        ],
        reasoning="".join(
            [str(info[agent]["comments"]) for agent in env.agents[:2]]
        ),  # Keep first 2 for compatibility
        rewards=[info[agent_name]["complete_rating"] for agent_name in env.agents],
        rewards_prompt=info["rewards_prompt"]["overall_prompt"],
    )
    print("Reward prompt: ")
    rich.print(epilog.rewards_prompt)
    agent_profiles, conversation = epilog.render_for_humans()
    print("Agent profiles: ")
    for agent_profile in agent_profiles:
        rich.print(agent_profile)
    for message in conversation:
        rich.print(message)

    if push_to_db:
        try:
            epilog.save()
        except Exception as e:
            logging.error(f"Failed to save episode log: {e}")
    # flatten nested list messages
    return list(itertools.chain(*messages))


async def aevaluate_one_episode(
    episode: EpisodeLog,
    model: str = "gpt-4",
    tag: str | None = None,
    push_to_db: bool = False,
) -> None:
    history = "\n".join(episode.render_for_humans()[1][:-2])
    evaluator: EpisodeLLMEvaluator[SotopiaDimensions] = EpisodeLLMEvaluator(
        model_name=model,
        response_format_class=EvaluationForAgents[SotopiaDimensions],
    )
    response = unweighted_aggregate_evaluate(
        list(
            itertools.chain(
                *await asyncio.gather(
                    *[
                        single_evaluator.__acall__(
                            turn_number=-1,
                            history=history,
                            messages=None,
                        )
                        for single_evaluator in [evaluator]
                    ]
                )
            )
        )
    )
    info: dict[str, dict[str, str | ScriptEnvironmentResponse | float | None]] = {}
    for i, agent_name in enumerate(episode.agents):
        # Try to find reward in various possible locations
        rating: float | tuple[float, dict[str, float]] | None = 0
        if response.rewards:
            # Check for direct name match or numeric index match (agent_1, agent_2...)
            if agent_name in response.rewards:
                rating = response.rewards[agent_name]
            elif f"agent_{i+1}" in response.rewards:
                rating = response.rewards[f"agent_{i+1}"]

        # Fallback to legacy p1_rate/p2_rate for 2-agent cases if rewards missing
        if rating == 0 or rating is None:
            if i == 0 and response.p1_rate is not None:
                rating = response.p1_rate
            elif i == 1 and response.p2_rate is not None:
                rating = response.p2_rate

        # Unpack tuple if necessary (value, metadata)
        if isinstance(rating, tuple):
            rating = rating[0]

        info[agent_name] = {
            "comments": response.comments or "",
            "complete_rating": rating or 0,
        }

    assert isinstance(episode.models, list)
    # Generic model list: [env_model, agent1_model, agent2_model...]
    log_models = [model] + episode.models[1:] if episode.models else [model]

    epilog = EpisodeLog(
        environment=episode.environment,
        agents=episode.agents,
        tag=tag,
        models=log_models,
        messages=episode.messages,
        reasoning="\n".join([str(info[agent]["comments"]) for agent in episode.agents]),
        rewards=[info[agent_name]["complete_rating"] for agent_name in episode.agents],
        rewards_prompt="TBD",
    )

    if push_to_db:
        try:
            epilog.save()
        except Exception as e:
            logging.error(f"Failed to save episode log: {e}")
