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
    seller = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Seller", AgentProfile.last_name == "One"
        ).all()[0],
    )
except (IndexError, NotImplementedError):
    print("Seller not found, creating new agent profile.")
    add_agent_to_database(
        first_name="Seller",
        last_name="One",
        age=45,
        occupation="Antique Dealer",
        gender="Woman",
        gender_pronoun="she/her",
        big_five="high extraversion, low neuroticism",
        moral_values=["fairness", "honesty"],
        decision_making_style="strategic",
        secret="The item cost me $80, I need at least $120 profit",
    )
    seller = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Seller", AgentProfile.last_name == "One"
        ).all()[0],
    )

try:
    buyer1 = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Buyer", AgentProfile.last_name == "One"
        ).all()[0],
    )
except (IndexError, NotImplementedError):
    print("Buyer One not found, creating new agent profile.")
    add_agent_to_database(
        first_name="Buyer",
        last_name="One",
        age=30,
        occupation="Collector",
        gender="Man",
        gender_pronoun="he/him",
        big_five="high openness, high conscientiousness",
        moral_values=["value", "respect"],
        decision_making_style="analytical",
        secret="I have a budget of $250 but want to get the best deal",
    )
    buyer1 = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Buyer", AgentProfile.last_name == "One"
        ).all()[0],
    )


# Add the second buyer back
try:
    buyer2 = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Buyer", AgentProfile.last_name == "Two"
        ).all()[0],
    )
except (IndexError, NotImplementedError):
    print("Buyer Two not found, creating new agent profile.")
    add_agent_to_database(
        first_name="Buyer",
        last_name="Two",
        age=28,
        occupation="Reseller",
        gender="Non-binary",
        gender_pronoun="they/them",
        big_five="high agreeableness, moderate extraversion",
        moral_values=["cooperation", "profit"],
        decision_making_style="competitive",
        secret="I can resell this for $300, willing to pay up to $180",
    )
    buyer2 = cast(
        AgentProfile,
        AgentProfile.find(
            AgentProfile.first_name == "Buyer", AgentProfile.last_name == "Two"
        ).all()[0],
    )

# Scenario for three-party negotiation
scenario = """Antique auction negotiation: A seller has a rare vintage item. Two buyers are competing to purchase it.
The seller wants to maximize profit while the buyers want to get the best deal.
Buyers can outbid each other or collaborate to drive the price down."""

# Goals for each agent
seller_goal = """You are selling a rare vintage clock. You want to:
1. Get at least $200 for the item (it cost you $80)
2. Create competition between the two buyers
3. Close the deal within 10 rounds of negotiation
Start by describing the item and asking for initial offers."""

buyer1_goal = """You are interested in buying a vintage clock for your collection. You want to:
1. Pay no more than $250 (your maximum budget)
2. Get the item for the best possible price
3. Either outbid the other buyer or convince them to withdraw
You value authentic pieces and are willing to pay fair prices."""

buyer2_goal = """You want to buy the vintage clock to resell it. You want to:
1. Pay no more than $180 (to ensure profit margin)
2. Either win the bid or collaborate with the other buyer
3. Make strategic offers to test the seller's limits
You know the market value is around $300."""


add_env_profile(scenario=scenario, agent_goals=[seller_goal, buyer1_goal, buyer2_goal])

# Get the most recently saved environment
all_envs = list(EnvironmentProfile.find().all())
last_env = cast(EnvironmentProfile, all_envs[-1])

# Build a sampler with one seller and two buyers
sampler: UniformSampler[Observation, AgentAction] = UniformSampler[
    Observation, AgentAction
](env_candidates=[last_env], agent_candidates=[seller, buyer1, buyer2])


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
