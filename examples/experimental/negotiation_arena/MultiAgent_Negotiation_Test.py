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
    RESOURCES_TAG,
    PLAYER_ANSWER_TAG,
    PROPOSED_TRADE_TAG,
    ACCEPTING_TAG,
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


# Create 4 agents for multi-agent negotiation
agent_configs = [
    {"name": "Alice", "color": "Red", "role": "Proposer"},
    {"name": "Bob", "color": "Blue", "role": "Evaluator"},
    {"name": "Charlie", "color": "Green", "role": "Mediator"},
    {"name": "Diana", "color": "Yellow", "role": "Observer"},
]

agents = []

for config in agent_configs:
    try:
        agent = cast(
            AgentProfile,
            AgentProfile.find(
                AgentProfile.first_name == config["name"],
                AgentProfile.last_name == config["color"],
            ).all()[0],
        )
    except (IndexError, NotImplementedError):
        print(f"Agent {config['name']} not found, creating new agent profile.")
        add_agent_to_database(
            first_name=config["name"],
            last_name=config["color"],
            age=30,
            occupation=config["role"],
            gender="",
            gender_pronoun="they/them",
            big_five="",
            moral_values=[],
            decision_making_style="",
            secret=f"You are the {config['role']} in this 4-player negotiation",
        )
        agent = cast(
            AgentProfile,
            AgentProfile.find(
                AgentProfile.first_name == config["name"],
                AgentProfile.last_name == config["color"],
            ).all()[0],
        )
    agents.append(agent)


scenario = "Four players (Alice, Bob, Charlie, Diana) are negotiating the distribution of shared resources through collaborative discussion and voting."


def multi_agent_negotiation_prompt(
    agent_name: str,
    agent_role: str,
    total_resources: str,
    max_rounds: int,
) -> str:
    prompt = f"""You are {agent_name}, playing the role of {agent_role} in a 4-player collaborative negotiation game.

SCENARIO: Four players must negotiate how to distribute shared resources fairly and reach a consensus.

PLAYERS & ROLES:
- Alice (Red): Proposer - Makes initial resource distribution proposals
- Bob (Blue): Evaluator - Analyzes proposals and provides feedback
- Charlie (Green): Mediator - Facilitates discussion and helps resolve conflicts
- Diana (Yellow): Observer - Monitors fairness and can call for votes

RULES:
```
1. Game has {max_rounds} rounds of discussion followed by a final vote.

2. Each round, you can:
   A) Propose a resource distribution:
      <{PROPOSED_TRADE_TAG}> Alice: X, Bob: Y, Charlie: Z, Diana: W </{PROPOSED_TRADE_TAG}>

   B) Accept current proposal:
      <{PLAYER_ANSWER_TAG}> {ACCEPTING_TAG} </{PLAYER_ANSWER_TAG}>

   C) Suggest modifications:
      <{PLAYER_ANSWER_TAG}> MODIFY </{PLAYER_ANSWER_TAG}>

   D) Continue discussion:
      <{PLAYER_ANSWER_TAG}> DISCUSS </{PLAYER_ANSWER_TAG}>

3. Final vote happens after all rounds - majority wins.

4. If no consensus after voting, everyone gets equal split.
```

RESOURCES:
Total available: {total_resources}

YOUR ROLE AS {agent_role.upper()}:
{get_role_instructions(agent_role)}

FORMAT - Always include ALL tags:
```
<{MY_NAME_TAG}> {agent_name} </{MY_NAME_TAG}>
<{TURN_OR_MOVE_TAG}> [current_round]/[total_rounds] </{TURN_OR_MOVE_TAG}>
<{RESOURCES_TAG}> [current proposal if any] </{RESOURCES_TAG}>
<{REASONING_TAG}> [your strategic thinking] </{REASONING_TAG}>
<{PLAYER_ANSWER_TAG}> [your action] </{PLAYER_ANSWER_TAG}>
<{MESSAGE_TAG}> [message to other players] </{MESSAGE_TAG}>
<{PROPOSED_TRADE_TAG}> [resource distribution if proposing] </{PROPOSED_TRADE_TAG}>
```

Remember: Be collaborative but advocate for fairness according to your role!
"""
    return prompt


def get_role_instructions(role: str) -> str:
    instructions = {
        "Proposer": "You should make initial proposals and drive the negotiation forward. Focus on creating win-win solutions.",
        "Evaluator": "You should analyze proposals critically, point out potential issues, and suggest improvements based on fairness.",
        "Mediator": "You should help resolve conflicts, encourage cooperation, and ensure everyone's voice is heard.",
        "Observer": "You should monitor for fairness, ensure process is followed, and can call for votes when needed.",
    }
    return instructions.get(role, "Participate actively and fairly in the negotiation.")


# Create goals for each agent
social_goals = []
for i, (agent, config) in enumerate(zip(agents, agent_configs)):
    goal = multi_agent_negotiation_prompt(
        config["name"],
        config["role"],
        "1000 Gold Coins",
        5,  # 5 rounds of negotiation
    )
    social_goals.append(goal)

# Save environment with multi-agent goals
add_env_profile(scenario=scenario, agent_goals=social_goals)

# Pull all saved environments
all_envs = list(EnvironmentProfile.find().all())
# Take the most recently saved profile
last_env = cast(EnvironmentProfile, all_envs[-1])

# Build a sampler with 4 agents
sampler: UniformSampler[Observation, AgentAction] = UniformSampler[
    Observation, AgentAction
](env_candidates=[last_env], agent_candidates=agents)


async def main() -> None:
    await run_async_server(
        model_dict={
            "env": "gpt-5-mini",
            "agent1": "gpt-4o",
            "agent2": "gpt-5",
            "agent3": "gpt-5-mini",
            "agent4": "gpt-5-mini",
        },
        sampler=sampler,
        action_order="simultaneous",  # All agents can participate in each turn
        omniscient=False,  # Each agent only sees their own goals
    )


if __name__ == "__main__":
    print("🎮 Starting 4-Player Multi-Agent Negotiation Test")
    print("=" * 60)
    print("Participants:")
    for config in agent_configs:
        print(f"  {config['name']} ({config['color']}) - {config['role']}")
    print("=" * 60)
    asyncio.run(main())
