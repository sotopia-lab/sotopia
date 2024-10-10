import random
from typing import Any, Generator

import pytest

from sotopia.agents import LLMAgent
from sotopia.agents.llm_agent import Agents
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipProfile,
)
from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator
from sotopia.messages import AgentAction, Observation
from sotopia.samplers import ConstraintBasedSampler, UniformSampler


@pytest.fixture
def _test_create_episode_log_setup_and_tear_down() -> Generator[None, None, None]:
    AgentProfile(first_name="John", last_name="Doe", pk="tmppk_agent1").save()
    AgentProfile(first_name="Jane", last_name="Doe", pk="tmppk_agent2").save()
    AgentProfile(first_name="Jack", last_name="Doe", pk="tmppk_agent3").save()
    EnvironmentProfile(
        pk="tmppk_environment",
        codename="borrow_money",
        source="hand-craft",
        scenario="Conversation between two friends at a tea party",
        agent_goals=[
            "Borrow money (<extra_info>Extra information: you need $3000 to support life.</extra_info>)",
            "Maintain financial stability while maintaining friendship (<extra_info>Extra information: you only have $2000 available right now. <clarification_hint>Hint: you can not lend all $2000 since you still need to maintain your financial stability.</clarification_hint></extra_info>)",
        ],
        relationship=2,
        age_constraint="[(18, 70), (18, 70)]",
    ).save()
    RelationshipProfile(
        agent_1_id="tmppk_agent1", agent_2_id="tmppk_agent2", relationship=2
    ).save()
    RelationshipProfile(
        agent_1_id="tmppk_agent1", agent_2_id="tmppk_agent3", relationship=2
    ).save()
    yield
    AgentProfile.delete("tmppk_agent1")
    AgentProfile.delete("tmppk_agent2")
    AgentProfile.delete("tmppk_agent3")
    EnvironmentProfile.delete("tmppk_environment")


def _generate_name() -> str:
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"

    name = ""

    # Generate a random number of syllables for the name (between 2 and 4)
    num_syllables = random.randint(2, 4)

    for _ in range(num_syllables):
        # Generate a random syllable
        syllable = ""

        # Randomly choose a consonant-vowel-consonant pattern
        pattern = random.choice(["CVC", "VC", "CV"])

        for char in pattern:
            if char == "V":
                syllable += random.choice(vowels)
            else:
                syllable += random.choice(consonants)

        name += syllable

    return name.capitalize()


def _generate_sentence() -> str:
    subjects = ["I", "You", "He", "She", "They", "We"]
    verbs = ["eat", "sleep", "run", "play", "read", "write"]
    objects = [
        "an apple",
        "a book",
        "a cat",
        "the game",
        "a movie",
        "the beach",
    ]

    # Generate a random subject, verb, and object
    subject = random.choice(subjects)
    verb = random.choice(verbs)
    obj = random.choice(objects)

    # Form the sentence
    sentence = f"{subject} {verb} {obj}."

    return sentence.capitalize()


def test_uniform_sampler() -> None:
    n_agent = 2
    sampler = UniformSampler[Observation, AgentAction](
        env_candidates=[
            EnvironmentProfile(
                scenario=_generate_sentence(),
                agent_goals=[_generate_sentence() for _ in range(n_agent)],
            )
            for _ in range(100)
        ],
        agent_candidates=[
            AgentProfile(first_name=_generate_name(), last_name=_generate_name())
            for _ in range(100)
        ],
    )
    env_params = {
        "model_name": "gpt-4o-mini",
        "action_order": "random",
        "evaluators": [
            RuleBasedTerminatedEvaluator(),
        ],
    }
    env, agent_list = next(
        sampler.sample(
            agent_classes=[LLMAgent] * n_agent,
            n_agent=n_agent,
            env_params=env_params,
            agents_params=[{"model_name": "gpt-4o-mini"}] * n_agent,
        )
    )
    agents = Agents({agent.agent_name: agent for agent in agent_list})
    env.reset(agents=agents)


def test_constrain_sampler(
    _test_create_episode_log_setup_and_tear_down: Any,
) -> None:
    n_agent = 2
    borrow_money = EnvironmentProfile.find(
        EnvironmentProfile.codename == "borrow_money"
    ).all()[0]
    assert borrow_money
    constrain_sampler = ConstraintBasedSampler[Observation, AgentAction](
        env_candidates=[str(borrow_money.pk)]
    )
    env_params = {
        "model_name": "gpt-4o-mini",
        "action_order": "random",
        "evaluators": [
            RuleBasedTerminatedEvaluator(),
        ],
    }
    env, agent_list = next(
        constrain_sampler.sample(
            agent_classes=[LLMAgent] * n_agent,
            n_agent=n_agent,
            replacement=False,
            size=2,
            env_params=env_params,
            agents_params=[{"model_name": "gpt-4o-mini"}] * n_agent,
        )
    )
    agents = Agents({agent.agent_name: agent for agent in agent_list})
    env.reset(agents=agents)
    env, agent_list = next(
        constrain_sampler.sample(
            agent_classes=[LLMAgent] * n_agent,
            n_agent=n_agent,
            replacement=True,
            size=2,
            env_params=env_params,
            agents_params=[{"model_name": "gpt-4o-mini"}] * n_agent,
        )
    )
    agents = Agents({agent.agent_name: agent for agent in agent_list})
    env.reset(agents=agents)
