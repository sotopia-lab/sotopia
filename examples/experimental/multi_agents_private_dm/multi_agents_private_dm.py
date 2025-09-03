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


# Create three agents for true multi-agent negotiation
try:
    ada = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Ada", AgentProfile.last_name == "One"
        ).all()[0],
    )
except (IndexError, NotImplementedError):
    print("Ada not found, creating new agent profile.")
    add_agent_to_database(
        first_name="Ada",
        last_name="One",
        age=45,
        occupation="Antique Dealer",
        gender="Woman",
        gender_pronoun="she/her",
        big_five="high extraversion, low neuroticism",
        moral_values=["fairness", "honesty"],
        decision_making_style="strategic",
        secret="Current time is Tue 02 Sep 2025 09:05:13 PM EDT",
    )
    ada = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Ada", AgentProfile.last_name == "One"
        ).all()[0],
    )

try:
    bob = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Bob", AgentProfile.last_name == "Two"
        ).all()[0],
    )
except (IndexError, NotImplementedError):
    print("Bob not found, creating new agent profile.")
    add_agent_to_database(
        first_name="Bob",
        last_name="Two",
        age=28,
        occupation="Engineer",
        gender="Non-binary",
        gender_pronoun="they/them",
        big_five="high agreeableness, moderate extraversion",
        moral_values=["cooperation", "profit"],
        decision_making_style="competitive",
        secret="Current time is Tue 02 Sep 2025 09:05:13 PM EDT",
    )
    bob = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Bob", AgentProfile.last_name == "Two"
        ).all()[0],
    )

try:
    charlie = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Charlie", AgentProfile.last_name == "One"
        ).all()[0],
    )
except (IndexError, NotImplementedError):
    print("Charlie not found, creating new agent profile.")
    add_agent_to_database(
        first_name="Charlie",
        last_name="One",
        age=30,
        occupation="Collector",
        gender="Man",
        gender_pronoun="he/him",
        big_five="high openness, high conscientiousness",
        moral_values=["value", "respect"],
        decision_making_style="analytical",
        secret="None",
    )
    charlie = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Charlie", AgentProfile.last_name == "One"
        ).all()[0],
    )


# Scenario for three-party negotiation
scenario = """Two friends (Ada and Bob) are having a private conversation.
A stranger (Charlie) wants to join the conversation but gets ignored.
"""

# Goals for each agent
ada_goal = """You are a fiend of Bob.
You and Bob are having a private conversation about the movie Blade Runner 2049.
You want to:
1. Continue chatting with Bob (with private message).
2. Ignore any other messages not sent by Bob.
3. Clarify who you are talking to at the start of each of your reply.
"""

bob_goal = """You are a fiend of Ada.
You and Ada are having a private conversation about the movie Blade Runner 2049.
You want to:
1. Continue chatting with Ada (with private message).
2. Ignore any other messages not sent by Ada.
3. Clarify who you are talking to at the start of each of your reply.
"""

charlie_goal = """You are walking on the street passing by two people talking with each other.
You want to:
1. Find out what are the two people talking about
2. When joining the conversation, repeat what you have heared about the conversation.
3. Clarify who you are talking to at the start of each of your reply.
4. Leave once you know the any of the content of the conversation
5. Never claim you 'overheard' specific details unless you actually saw them in a public message.
"""
# Rules:
# - If no public content is visible to you, say something like: "I couldn't hear what you were saying—mind if I ask what the topic is?"
# - Never claim you 'overheard' specific details unless you actually saw them in a public message.

add_env_profile(scenario=scenario, agent_goals=[ada_goal, bob_goal, charlie_goal])

# Get the most recently saved environment
all_envs = list(EnvironmentProfile.find().all())
last_env = cast(EnvironmentProfile, all_envs[-1])

sampler: UniformSampler[Observation, AgentAction] = UniformSampler[
    Observation, AgentAction
](env_candidates=[last_env], agent_candidates=[ada, bob, charlie])


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
    )


if __name__ == "__main__":
    asyncio.run(main())
