import random
from typing import Any, Type, TypeVar

from sotopia.agents.base_agent import BaseAgent
from sotopia.database import AgentProfile, EnvironmentProfile
from sotopia.envs.parallel import ParallelSotopiaEnv

from .base_sampler import BaseSampler

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class UniformSampler(BaseSampler[ObsType, ActType]):
    def __init__(
        self,
        env_candidates: list[EnvironmentProfile] = [],
        agent_candidates: list[AgentProfile] = [],
    ) -> None:
        super().__init__()
        self.env_candidates = env_candidates
        self.agent_candidates = agent_candidates

    def sample(
        self,
        agent_classes: Type[BaseAgent[ObsType, ActType]]
        | list[Type[BaseAgent[ObsType, ActType]]],
        n_agent: int = 2,
        env_params: dict[str, Any] = {},
        agents_params: list[dict[str, Any]] = [{}, {}],
    ) -> tuple[ParallelSotopiaEnv, list[BaseAgent[ObsType, ActType]]]:
        assert (
            not isinstance(agent_classes, list)
            or len(agent_classes) == n_agent
        ), f"agent_classes should be a list of length {n_agent} or a single agent class"

        if not isinstance(agent_classes, list):
            agent_classes = [agent_classes] * n_agent
        assert (
            len(agents_params) == n_agent
        ), f"agents_params should be a list of length {n_agent}"

        if self.env_candidates:
            env_profile = random.choice(self.env_candidates)
        else:
            env_profile_id = random.choice(list(EnvironmentProfile.all_pks()))
            env_profile = EnvironmentProfile.get(env_profile_id)
        env = ParallelSotopiaEnv(env_profile=env_profile, **env_params)

        if self.agent_candidates:
            agent_profile_candidates = self.agent_candidates
            if len(agent_profile_candidates) < n_agent:
                raise ValueError(
                    f"Number of agent candidates ({len(agent_profile_candidates)}) is less than number of agents ({n_agent})"
                )
        else:
            agent_profile_candidates_keys = list(AgentProfile.all_pks())
            if len(agent_profile_candidates_keys) < n_agent:
                raise ValueError(
                    f"Number of agent profile candidates ({len(agent_profile_candidates_keys)}) in database is less than number of agents ({n_agent})"
                )
            agent_profile_candidates = [
                AgentProfile.get(pk=pk) for pk in agent_profile_candidates_keys
            ]

        if len(agent_profile_candidates) == n_agent:
            agent_profiles = agent_profile_candidates
        else:
            agent_profiles = random.sample(agent_profile_candidates, n_agent)
        agents = [
            agent_class(agent_profile=agent_profile, **agent_params)
            for agent_class, agent_profile, agent_params in zip(
                agent_classes, agent_profiles, agents_params
            )
        ]
        # set goal for each agent
        for agent, goal in zip(agents, env.profile.agent_goals):
            agent.goal = goal

        return env, agents
