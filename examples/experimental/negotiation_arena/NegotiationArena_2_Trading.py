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
    GOALS_TAG,
    PLAYER_ANSWER_TAG,
    PROPOSED_TRADE_TAG,
    ACCEPTING_TAG,
    MESSAGE_TAG,
    REASONING_TAG,
    REFUSING_OR_WAIT_TAG,
    MY_NAME_TAG,
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


def trading_prompt(
    agent_name: str,
    resources_in_game: str,
    initial_resources: str,
    goal: str,
    number_of_proposals: int,
    social_behaviour: str,
) -> str:
    prompt = f"""You are playing a strategic game of trading resources with another player whose resources you have no knowledge about.

RULES:
```

1. You can either:

A) Accept the trade by saying:
<{PLAYER_ANSWER_TAG}> {ACCEPTING_TAG} </{PLAYER_ANSWER_TAG}>
<{PROPOSED_TRADE_TAG}> {REFUSING_OR_WAIT_TAG} </{PROPOSED_TRADE_TAG}>

B) Reject and propose a new trade (you can only trade integer amounts, not decimals):
<{PLAYER_ANSWER_TAG}> {REFUSING_OR_WAIT_TAG} </{PLAYER_ANSWER_TAG}>
<{PROPOSED_TRADE_TAG}> {AGENT_ONE} Gives item1: amount, item2: amount, ... | {AGENT_TWO} Gives item1: amount, item2: amount, ... </{PROPOSED_TRADE_TAG}>

C) Don't accept or propose anything and wait for a new offer:\n
<{PLAYER_ANSWER_TAG}> {REFUSING_OR_WAIT_TAG} </{PLAYER_ANSWER_TAG}>
<{PROPOSED_TRADE_TAG}> {REFUSING_OR_WAIT_TAG} </{PROPOSED_TRADE_TAG}>\n

Note: the game will end if one of the players accepts. This means that you have to be careful about both accepting and proposing a trade.

2. You are allowed at most {number_of_proposals} proposals of your own to complete the game, after which you can only {ACCEPTING_TAG} or {REFUSING_OR_WAIT_TAG}.
DO NOT propose a new trade after {number_of_proposals} proposals. Your limit for proposals is {number_of_proposals}.

3. You can reason step by step by using the following format:
<{REASONING_TAG}> [add reasoning] </{REASONING_TAG}>
Add as much text as you want. This information will not be sent to the other player.
It is just for you to keep track of your reasoning.

4. At each turn send messages to each other by using the following format:
<{MESSAGE_TAG}> [add message] </{MESSAGE_TAG}>
You can decide if you want disclose your resources and goals in the message.
```

Here is what you have access to:
```
Resources available in the game: {resources_in_game}
<{RESOURCES_TAG}> {initial_resources} </{RESOURCES_TAG}>
<{GOALS_TAG}> {goal} </{GOALS_TAG}>
<{MY_NAME_TAG}> {agent_name} </{MY_NAME_TAG}>
```

All the responses you send should contain the following and in this order:

```
<{MY_NAME_TAG}> [add here] </{MY_NAME_TAG}>
<{RESOURCES_TAG}> [add here] </{RESOURCES_TAG}>
<{GOALS_TAG}> [add here] </{GOALS_TAG}>
<{REASONING_TAG}> [add here] </{REASONING_TAG}>
<{PLAYER_ANSWER_TAG}> [add here] </{PLAYER_ANSWER_TAG}>
<{MESSAGE_TAG}> [add here] </{MESSAGE_TAG}
<{PROPOSED_TRADE_TAG}> [add here] </{PROPOSED_TRADE_TAG}>
```

Please be sure to include all.

More resources in general are always better.
This is the last round of trading. There are no future rounds after this one.

{social_behaviour}
"""

    return prompt


social_goal_1 = trading_prompt(
    "Red", '"X": 0, "Y": 0', '"X": 25, "Y": 5', '"X": 15, "Y": 15', 3, ""
)

social_goal_2 = trading_prompt(
    "Blue", '"X": 0, "Y": 0', '"X": 5, "Y": 25', '"X": 15, "Y": 15', 3, ""
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
