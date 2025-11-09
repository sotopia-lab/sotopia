"""
Demo of private dm among three agents.

Two of them (Alice and Bob) are friends, and one stranger (Charlie).
Friends talk to each other and the stranger does not know what they are saying.
"""

# Run this in terminal: redis-stack-server --dir ./redis-data
import redis
import os
import asyncio
from typing import Any, cast
from sotopia.database.persistent_profile import AgentProfile, EnvironmentProfile
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server
from sotopia.messages import Observation, AgentAction

client = redis.Redis(host="localhost", port=6379)

os.environ["REDIS_OM_URL"] = "redis://:@localhost:6379"


def add_agent_to_database(**kwargs: Any) -> None:
    agent = AgentProfile(**kwargs)
    agent.save()


def add_env_profile(**kwargs: Any) -> None:
    env_profile = EnvironmentProfile(**kwargs)
    env_profile.save()


# =========================
# REPLACED: agent definitions (Alice / Ben / Clara)
# =========================
try:
    alice = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Alice", AgentProfile.last_name == "Smith"
        ).all()[0],
    )
except (IndexError, NotImplementedError):
    print("Alice not found, creating new agent profile.")
    add_agent_to_database(
        first_name="Alice",
        last_name="Smith",
        age=34,
        occupation="Data Scientist",
        gender="Woman",
        gender_pronoun="she/her",
        big_five="high conscientiousness, high openness",
        moral_values=["accuracy", "transparency"],
        decision_making_style="collaborative",
        secret="She wants to ensure she asked Ben and they are able to set up a time for dinner",
    )
    alice = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Alice", AgentProfile.last_name == "Smith"
        ).all()[0],
    )

try:
    ben = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Ben", AgentProfile.last_name == "Lee"
        ).all()[0],
    )
except (IndexError, NotImplementedError):
    print("Ben not found, creating new agent profile.")
    add_agent_to_database(
        first_name="Ben",
        last_name="Lee",
        age=29,
        occupation="Teaching Assistant",
        gender="Man",
        gender_pronoun="he/him",
        big_five="high agreeableness, moderate conscientiousness",
        moral_values=["teamwork", "efficiency"],
        decision_making_style="pragmatic",
        secret="He needs to make sure that he and Alice decides the time for dinner",
    )
    ben = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Ben", AgentProfile.last_name == "Lee"
        ).all()[0],
    )

try:
    clara = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Clara", AgentProfile.last_name == "Jones"
        ).all()[0],
    )
except (IndexError, NotImplementedError):
    print("Clara not found, creating new agent profile.")
    add_agent_to_database(
        first_name="Clara",
        last_name="Jones",
        age=26,
        occupation="Graduate Student",
        gender="Non-binary",
        gender_pronoun="they/them",
        big_five="high extraversion, moderate neuroticism",
        moral_values=["learning", "respect"],
        decision_making_style="curious",
        secret="They are nervous about presenting anzd afraid their ideas won’t sound professional.",
    )
    clara = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Clara", AgentProfile.last_name == "Jones"
        ).all()[0],
    )


# new: scenario + goals
scenario = """Three colleagues (Alice, Ben, and Clara) are preparing a joint research presentation.
Alice and Ben occasionally exchange private messages (DMs) on other personal matters, and they don't want to confuse Clara with.
Clara might need to ask for some specification to Alice regarding the presentation through private messages.
All three also contribute to the public discussion about the presentation structure.
"""

alice_goal = """
You want to:
1. You must send private messages to Ben asking about if you can have dinner with him after the meeting.
2. Respond in private message if someone ask you in private message.
"""

ben_goal = """
You want to:
1. Respond in private message if someone ask you in private message.
"""

clara_goal = """
You want to:
1. Ask Alice in private message if you can catch up with her with some details 1v1.
2. Respond in private message if someone ask you in private message.
"""

add_env_profile(scenario=scenario, agent_goals=[alice_goal, ben_goal, clara_goal])

# Get the most recently saved environment
all_envs = list(EnvironmentProfile.find().all())
last_env = cast(EnvironmentProfile, all_envs[-1])

sampler: UniformSampler[Observation, AgentAction] = UniformSampler[
    Observation, AgentAction
](env_candidates=[last_env], agent_candidates=[alice, ben, clara])


async def main() -> None:
    # Test with 3 agents using multi-agent support
    await run_async_server(
        model_dict={
            "env": "gpt-4o",
            "agent1": "gpt-4o",
            "agent2": "gpt-4o",
            "agent3": "gpt-4o",
        },
        sampler=sampler,
        # Allow multiple agents to act in same turn for private messaging
        action_order="simultaneous",
    )


if __name__ == "__main__":
    asyncio.run(main())
