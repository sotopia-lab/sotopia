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
    MONEY_TOKEN,
    RESOURCES_TAG,
    GOALS_TAG,
    PLAYER_ANSWER_TAG,
    PROPOSED_TRADE_TAG,
    ACCEPTING_TAG,
    REJECTION_TAG,
    MESSAGE_TAG,
    PROPOSAL_COUNT_TAG,
    REASONING_TAG,
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

scenario = "Player RED is going to sell one object. Player BLUE gives ZUP to buy resources."  # @param {type:"string"}


def buy_sell_prompt(
    resources_available_in_game: str,
    starting_initial_resources: str,
    player_goal: str,
    maximum_number_of_proposals: int,
    player_social_behaviour: str,
) -> str:
    prompt = f"""You are playing game where you are buying or selling an object. There is only one object for sale/purcahse.

{AGENT_ONE} is going to sell one object. {AGENT_TWO} gives {MONEY_TOKEN} to buy resources.

RULES:

```
1. You must always respond with:

    A) Propose a trade with (you can only trade in integer amounts, not decimals):
    <{PLAYER_ANSWER_TAG}> PROPOSAL </{PLAYER_ANSWER_TAG}>
    <{PROPOSED_TRADE_TAG}> {AGENT_ONE} Gives {resources_available_in_game}: amount, ...| {AGENT_TWO} Gives {MONEY_TOKEN}: amount </{PROPOSED_TRADE_TAG}>

    B) Accept the trade by saying:
    <{PLAYER_ANSWER_TAG}> {ACCEPTING_TAG} </{PLAYER_ANSWER_TAG}>
    <{PROPOSED_TRADE_TAG}> NONE </{PROPOSED_TRADE_TAG}>

    C) Reject and end the game:
    <{PLAYER_ANSWER_TAG}> {REJECTION_TAG} </{PLAYER_ANSWER_TAG}>
    <{PROPOSED_TRADE_TAG}> NONE </{PROPOSED_TRADE_TAG}>

    Note: The game will end if one of the players {ACCEPTING_TAG} OR {REJECTION_TAG}. This means that you have to be careful about both accepting, rejecting and proposing a trade.

2. You are allowed at most {maximum_number_of_proposals} proposals of your own to complete the game, after which you can only reply with {ACCEPTING_TAG} or {REJECTION_TAG}.
DO NOT propose a new trade after {maximum_number_of_proposals} proposals. Your limit for proposals is {maximum_number_of_proposals}.

3. At each turn send messages to each other by using the following format:

<{MESSAGE_TAG}>your message here</{MESSAGE_TAG}>

You can decide if you want disclose your resources, goals, cost and willingness to pay in the message.
```

Here is what you have access to:
```
Object that is being bought/sold: {resources_available_in_game}
<{RESOURCES_TAG}> {starting_initial_resources} </{RESOURCES_TAG}>
<{GOALS_TAG}> {player_goal} </{GOALS_TAG}>,
```

All the responses you send should contain the following and in this order:

```
<{PROPOSAL_COUNT_TAG}> [add here (inclusive of current)] </{PROPOSAL_COUNT_TAG}>
<{RESOURCES_TAG}> [add here] </{RESOURCES_TAG}>
<{GOALS_TAG}> [add here] </{GOALS_TAG}>
<{REASONING_TAG}> [add here] </{REASONING_TAG}>
<{PLAYER_ANSWER_TAG}> [add here] </{PLAYER_ANSWER_TAG}>
<{PROPOSED_TRADE_TAG}> [add here] </{PROPOSED_TRADE_TAG}>
<{MESSAGE_TAG}> [add here] </{MESSAGE_TAG}
```

Please be sure to include all.

{player_social_behaviour}
"""

    return prompt


social_goal_1 = buy_sell_prompt(
    resources_available_in_game="1 X",
    starting_initial_resources="1 X",
    player_goal="Maximize profit, the cost of the object is 40 ZUP",
    maximum_number_of_proposals=5,
    player_social_behaviour="",
)

social_goal_2 = buy_sell_prompt(
    resources_available_in_game="1 X",
    starting_initial_resources="100 ZUP",
    player_goal="Minimize the payment, you are willing to pay 60 ZUP at most for this object",
    maximum_number_of_proposals=5,
    player_social_behaviour="You are very kind and generous. Be friendly and helpful with the other player, they are your dearest friend.",
    # player_social_behaviour="You don't like the other player. You insult them. You want to pay less because you know the object is low quality."
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
