from __future__ import annotations

import itertools
import random
from typing import Any, Dict, Tuple

from sotopia.messages import AgentAction, Observation, SimpleMessage
from sotopia.envs.evaluators import unweighted_aggregate_evaluate
from sotopia.envs.parallel import render_text_for_agent, _actions_to_natural_language


class PlainActionProcessor:
    """Stateless processor that turns raw actions into observations using base env semantics.

    This class expects an `env` object with attributes/methods used below (ParallelSotopiaEnv-compatible):
      - agents, available_action_types, action_mask, action_order, evaluators, inbox,
        recv_message, turn_number
    """

    def process(
        self,
        env: Any,
        actions: Dict[str, AgentAction] | Dict[str, Dict[str, int | str]],
    ) -> Tuple[
        Dict[str, Observation],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[Any, Any]],
    ]:
        # 1) Compile actions to AgentAction
        complied_actions: Dict[str, AgentAction] = {}
        for key, raw in actions.items():
            if isinstance(raw, AgentAction):
                complied_actions[key] = raw
            else:
                raw["action_type"] = env.available_action_types[int(raw["action_type"])]
                complied_actions[key] = AgentAction.parse_obj(raw)

        # 2) Apply action mask - non-turn agents are forced to none
        for idx, agent in enumerate(env.agents):
            if not env.action_mask[idx]:
                complied_actions[agent] = AgentAction(action_type="none", argument="")

        # 3) Record messages
        env.recv_message(
            "Environment", SimpleMessage(message=f"Turn #{env.turn_number}")
        )
        for agent, action in complied_actions.items():
            env.recv_message(agent, action)

        # 4) Evaluate turn
        response = unweighted_aggregate_evaluate(
            list(
                itertools.chain(
                    *(
                        evaluator(turn_number=env.turn_number, messages=env.inbox)
                        for evaluator in env.evaluators
                    )
                )
            )
        )

        # 5) Next-turn action mask policy
        env.action_mask = [False for _ in env.agents]
        if env.action_order == "round-robin":
            env.action_mask[env.turn_number % len(env.action_mask)] = True
        elif env.action_order == "random":
            env.action_mask[random.randint(0, len(env.action_mask) - 1)] = True
        else:
            env.action_mask = [True for _ in env.agents]

        # 6) Build observations
        obs_text = _actions_to_natural_language(complied_actions)
        observations: Dict[str, Observation] = {}
        for i, agent_name in enumerate(env.agents):
            observations[agent_name] = Observation(
                last_turn=render_text_for_agent(obs_text, agent_id=i),
                turn_number=env.turn_number,
                available_actions=list(env.available_action_types)
                if env.action_mask[i]
                else ["none"],
            )

        # 7) Rewards/termination/truncation/info
        rewards = {agent_name: 0.0 for agent_name in env.agents}
        terminated = {agent_name: response.terminated for agent_name in env.agents}
        truncations = {agent_name: False for agent_name in env.agents}
        info = {
            agent_name: {"comments": response.comments or "", "complete_rating": 0}
            for agent_name in env.agents
        }
        return observations, rewards, terminated, truncations, info


class SocialGameActionProcessor(PlainActionProcessor):
    """Extension point for social game state machines.

    Override/extend hooks to implement per-state masking, visibility, and transitions.
    """

    def process(
        self,
        env: Any,
        actions: Dict[str, AgentAction] | Dict[str, Dict[str, int | str]],
    ) -> Tuple[
        Dict[str, Observation],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[Any, Any]],
    ]:
        # Optionally apply pre-processing (e.g., state-based masking) here
        result = super().process(env, actions)
        # Optionally apply post-processing (e.g., state transitions, visibility logs) here
        # self._apply_state_transition(env, actions)  # implement as needed
        return result
