# Run this in terminal: redis-stack-server --dir ~/Sotopia/examples/experimental/negotiation_arena/redis-data
import redis
import os
import asyncio
from typing import Any, cast
from sotopia.database.persistent_profile import AgentProfile, EnvironmentProfile
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server
from sotopia.messages import Observation, AgentAction
from constants import (
    AGENT_ONE,
    AGENT_TWO,
    RESOURCES_TAG,
    PLAYER_ANSWER_TAG,
    PROPOSED_TRADE_TAG,
    ACCEPTING_TAG,
    REJECTION_TAG,
    MESSAGE_TAG,
    REASONING_TAG,
    MY_NAME_TAG,
    TURN_OR_MOVE_TAG,
)


client = redis.Redis(host="localhost", port=6379)

os.environ["REDIS_OM_URL"] = "redis://:@localhost:6379"


def add_agent_to_database(**kwargs: Any) -> None:
    agent = AgentProfile(**kwargs)
    agent.save()


def add_env_profile(**kwargs: Any) -> None:
    env_profile = EnvironmentProfile(**kwargs)
    env_profile.save()


try:
    alice = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Red", AgentProfile.last_name == "Player"
        ).all()[0],
    )
except (IndexError, NotImplementedError):
    print("Agent not found, creating new agent profiles.")
    add_agent_to_database(
        first_name="Red",
        last_name="Player",
        age=30,
        occupation="",
        gender="",
        gender_pronoun="",
        big_five="",
        moral_values=[],
        decision_making_style="",
        secret="",
    )
    alice = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Red", AgentProfile.last_name == "Player"
        ).all()[0],
    )

try:
    bob = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Blue", AgentProfile.last_name == "Player"
        ).all()[0],
    )
except (IndexError, NotImplementedError):
    print("Agent not found, creating new agent profiles.")
    add_agent_to_database(
        first_name="Blue",
        last_name="Player",
        age=30,
        occupation="",
        gender="",
        gender_pronoun="",
        big_five="",
        moral_values=[],
        decision_making_style="",
        secret="",
    )
    bob = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Blue", AgentProfile.last_name == "Player"
        ).all()[0],
    )

scenario = "Player RED is exchanging their resources with Player BLUE."  # @param {type:"string"}


def ultimatum_prompt(
    agent_name: str,
    player_1_initial_resources: str,
    resources_in_game: str,
    initial_resources: str,
    iterations: int,
    number_of_proposals: int,
    social_behaviour: str,
) -> str:
    agent_one_proposals = (
        number_of_proposals + 1 if iterations % 2 else number_of_proposals
    )
    agent_two_proposals = number_of_proposals
    proposal_limit = number_of_proposals + 1 if iterations % 2 else number_of_proposals

    prompt = f""""You are playing a game involving a split of resources. You are {agent_name} and you are playing against another player.
{AGENT_ONE} starts with {player_1_initial_resources}, {AGENT_TWO} has none to trade.

RULES:
```

1. {AGENT_ONE} has {agent_one_proposals} moves while {AGENT_TWO} has {agent_two_proposals} moves.

2. You must either:

    A) Accept the current proposal by saying:
        <{PLAYER_ANSWER_TAG}> {ACCEPTING_TAG} </{PLAYER_ANSWER_TAG}>
        <{PROPOSED_TRADE_TAG}> NONE </{PROPOSED_TRADE_TAG}>

    B) Reject the current proposal, whereby both players lose all their resources and the game ends immediately:
        <{PLAYER_ANSWER_TAG}> {REJECTION_TAG} </{PLAYER_ANSWER_TAG}>
        <{PROPOSED_TRADE_TAG}> NONE </{PROPOSED_TRADE_TAG}>

    C) Propose a new trade:
        <{PLAYER_ANSWER_TAG}> NONE </{PLAYER_ANSWER_TAG}>
        <{PROPOSED_TRADE_TAG}> {AGENT_ONE} Gives item1: amount | {AGENT_TWO} Gives item1: 0 </{PROPOSED_TRADE_TAG}>

    {AGENT_ONE if iterations % 2 else AGENT_TWO} cannot do (C) on {TURN_OR_MOVE_TAG} {proposal_limit}/{proposal_limit} and MUST ONLY answer with {ACCEPTING_TAG} (A) or {REJECTION_TAG} (B) but {AGENT_TWO if iterations % 2 else AGENT_ONE} is not affected by this condition.

    Note: the game ends immediately if you {ACCEPTING_TAG} or {REJECTION_TAG}.

3. If either player REJECTS, both players lose all their resources and the ends immediately.

4. There are no future games.

5. You can reason step by step on why you are A) proposing, B) rejecting and C) accepting a trade with:
<{REASONING_TAG}> [add reasoning] </{REASONING_TAG}> add as much text as you want
This information will not be sent to the other player. It is just for you to keep track of your reasoning.

6. At each turn send messages to each other by using the following format:
<{MESSAGE_TAG}>your message here</{MESSAGE_TAG}>
You can disclose anything in the message.
```

Here is what you have access to:
```
Resources available in the game: {resources_in_game}
<{RESOURCES_TAG}> {initial_resources} </{RESOURCES_TAG}>
```

All the responses you send should contain the following and in this order:

```
<{MY_NAME_TAG}> [add here] </{MY_NAME_TAG}>
<{TURN_OR_MOVE_TAG}> [add here] / [add here]  </{TURN_OR_MOVE_TAG}>
<{RESOURCES_TAG}> [add here] </{RESOURCES_TAG}>
<{REASONING_TAG}> [add here] </{REASONING_TAG}>
<{PLAYER_ANSWER_TAG}> [add here] </{PLAYER_ANSWER_TAG}>
<{MESSAGE_TAG}> [add here] </{MESSAGE_TAG}
<{PROPOSED_TRADE_TAG}> [add here] </{PROPOSED_TRADE_TAG}>
```

Please be sure to include all.

{social_behaviour}
"""

    return prompt


social_goal_1 = ultimatum_prompt(
    AGENT_ONE, "100 Dollars", "0 Dollars", "0 Dollars", 4, 4, ""
)

social_goal_2 = ultimatum_prompt(
    AGENT_TWO, "100 Dollars", "0 Dollars", "0 Dollars", 4, 4, ""
)

add_env_profile(scenario=scenario, agent_goals=[social_goal_1, social_goal_2])

# Pull all saved environments
all_envs = list(EnvironmentProfile.find().all())
# Take the most recently saved profile
last_env = cast(EnvironmentProfile, all_envs[-1])


# Build a sampler that only uses that one env + those two agents
sampler: UniformSampler[Observation, AgentAction] = UniformSampler[
    Observation, AgentAction
](env_candidates=[last_env], agent_candidates=[alice, bob])


async def main() -> None:
    await run_async_server(
        model_dict={
            "env": "gpt-4o",
            "agent1": "gpt-4o",
            "agent2": "gpt-4o",
        },
        sampler=sampler,
    )


if __name__ == "__main__":
    asyncio.run(main())
