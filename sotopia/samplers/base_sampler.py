from typing import Generic, Type, TypeVar

from sotopia.agents.base_agent import BaseAgent
from sotopia.envs.parallel import ParallelSotopiaEnv

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseSampler(Generic[ObsType, ActType]):
    def __init__(self) -> None:
        pass

    def sample(
        self,
        agent_classes: Type[BaseAgent[ObsType, ActType]]
        | list[Type[BaseAgent[ObsType, ActType]]],
        n_agent: int = 2,
    ) -> tuple[ParallelSotopiaEnv, list[BaseAgent[ObsType, ActType]]]:
        """Sample an environment and a list of agents.

        Args:
            agent_classes (Type[BaseAgent[ObsType, ActType]] | list[Type[BaseAgent[ObsType, ActType]]]): A single agent class for all sampled agents or a list of agent classes.
            n_agent (int, optional): Number of agents. Defaults to 2.

        Returns:
            tuple[ParallelSotopiaEnv, list[BaseAgent[ObsType, ActType]]]: an environment and a list of agents.
        """
        raise NotImplementedError
