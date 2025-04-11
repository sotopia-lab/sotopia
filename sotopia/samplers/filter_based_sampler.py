
import ast
import random
from typing import Any, Generator, Generic, Sequence, Type, TypeVar

from sotopia.agents.base_agent import BaseAgent
from sotopia.database import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipProfile,
)
from sotopia.envs.parallel import ParallelSotopiaEnv

from .base_sampler import BaseSampler, EnvAgentCombo

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

def age_filter(agent_id: str, age_constraint: str) -> bool:
    agent = AgentProfile.get(agent_id)
    age_constraint_list = ast.literal_eval(age_constraint)
    return (
        age_constraint_list[0][0]
        <= agent.age  # type: ignore[attr-defined]
        <= age_constraint_list[0][1]
    )
    
def occupation_filter(agent_id: str, occupation_constraint: str) -> bool:
    # TODO: handle the case where occupation_constraint == nan
    agent = AgentProfile.get(agent_id)
    occupation_constraint_list = ast.literal_eval(occupation_constraint)
    assert isinstance(occupation_constraint_list, list) or isinstance(occupation_constraint_list, str), "occupation_constraint should be a list or a string"
    if isinstance(occupation_constraint_list, str):
        return agent.occupation == occupation_constraint_list
    return agent.occupation == occupation_constraint

from typing import Callable, List
def filter_agents(filters: List[Callable[[str], bool]], agent_candidate_ids: List[str]) -> List[str]:
    return [agent_id for agent_id in agent_candidate_ids if all([filter(agent_id) for filter in filters])]

def _get_fit_agents_for_one_env(
    env_profile_id: str, agent_candidate_ids: list[set[str]] | None, size: int
) -> list[list[str]]:
    if agent_candidate_ids is None:
        print("agent_candidate_ids is None, using relationship")
        return _get_fit_agents_for_one_env_by_relationship(
            env_profile_id, agent_candidate_ids, size
        )
    else:
        # provide a list of agent ids
        print("agent_candidate_ids is not None, using candidate") # TODO change to logging
        return _get_fit_agents_for_one_env_by_candidate(
            env_profile_id, list([list(item) for item in agent_candidate_ids]), size
        )

def _get_fit_agents_for_one_env_by_candidate(
    env_profile_id: str, agent_candidate_ids: list[list[str]], size: int
) -> list[list[str]]:
    """
        NOTE: 
        1. In this setting we assume the relations are determined by scenarios and manually verified by human
        2. We only do random sampling w replacement so sometimes the same agent may appear multiple times
    """
    # 
    env = EnvironmentProfile.get(env_profile_id)
    age_constraint = env.age_constraint
    occupation_constraint = env.occupation_constraint
    age_filter_func = lambda agent_id: age_filter(agent_id, age_constraint)
    occupation_filter_func = lambda agent_id: occupation_filter(agent_id, occupation_constraint)
    
    fit_agents = agent_candidate_ids
    fit_agents_list = []
    # fit_agents = filter_agents([age_filter_func], agent_candidate_ids)
    # all_names = set([AgentProfile.get(pk).first_name + AgentProfile.get(pk).last_name for pk in fit_agents])
    
    # all_agents = [pk for pk in AgentProfile.all_pks() if pk not in fit_agents and AgentProfile.get(pk).occupation not in ["Job Hunter", "Recruiter"] and AgentProfile.get(pk).first_name + AgentProfile.get(pk).last_name not in all_names]
    
    # print("All names in fit_agents", all_names)
    # print("All names in all_agents", [AgentProfile.get(pk).first_name + AgentProfile.get(pk).last_name for pk in all_agents])

    for _ in range(size):
        while True:
            agents = [random.choice(agent_pool) for agent_pool in fit_agents]
            if AgentProfile.get(agents[0]).first_name + AgentProfile.get(agents[0]).last_name != AgentProfile.get(agents[1]).first_name + AgentProfile.get(agents[1]).last_name:
                break

        # if len(fit_agents) < 2:
        #     raise ValueError(
        #         f"Number of available agents ({len(fit_agents)}) "
        #         f"is smaller than the required size ({2})"
        #     )
        two_agents = agents
        fit_agents_list.append(two_agents)
    return fit_agents_list
    

def _get_fit_agents_for_one_env_by_relationship(
    env_profile_id: str, agent_candidate_ids: None, size: int
) -> list[list[str]]:
    env = EnvironmentProfile.get(env_profile_id)

    relationship_constraint = env.relationship
    available_relationships = RelationshipProfile.find(
        RelationshipProfile.relationship == relationship_constraint
    ).all()
    age_contraint = env.age_constraint
    assert isinstance(age_contraint, str)
    if age_contraint != "[(18, 70), (18, 70)]":
        age_contraint_list = ast.literal_eval(age_contraint)
        available_relationships = [
            relationship
            for relationship in available_relationships
            if (
                age_contraint_list[0][0]
                <= AgentProfile.get(relationship.agent_1_id).age  # type: ignore[attr-defined]
                <= age_contraint_list[0][1]
                and age_contraint_list[1][0]
                <= AgentProfile.get(relationship.agent_2_id).age  # type: ignore[attr-defined]
                <= age_contraint_list[1][1]
            )
        ]
    if len(available_relationships) < size:
        raise ValueError(
            f"Number of available relationships ({len(available_relationships)}) "
            f"is smaller than the required size ({size})"
        )
    random.shuffle(available_relationships)
    selected_relationship = available_relationships[:size]
    fit_agents = []
    for relationship in selected_relationship:
        assert isinstance(relationship, RelationshipProfile)
        fit_agents.append([relationship.agent_1_id, relationship.agent_2_id])
    return fit_agents

def filter_agent_ids(filter_funcs: List[Callable[[str], bool]], agent_candidate_ids: List[str]) -> List[set[str]]:
    return [set([agent_id for agent_id in agent_candidate_ids if filter_func(agent_id)]) for filter_func in filter_funcs]

class FilterBasedSampler(BaseSampler[ObsType, ActType]):
    def __init__(
        self,
        env_candidates: Sequence[EnvironmentProfile | str] | None = None,
        agent_candidates: Sequence[AgentProfile | str] | None = None,
        filter_func: List[Callable[[str], bool]] = [lambda agent_id: True],
    ) -> None:
        super().__init__(env_candidates, agent_candidates)
        self.filter_func = filter_func
        
    
    def sample(
        self,
        agent_classes: Type[BaseAgent[ObsType, ActType]]
        | list[Type[BaseAgent[ObsType, ActType]]],
        n_agent: int = 2,
        replacement: bool = True,
        size: int = 10,
        env_params: dict[str, Any] = {},
        agents_params: list[dict[str, Any]] = [{}, {}],
    ) -> Generator[EnvAgentCombo[ObsType, ActType], None, None]:
        """
        Sample an environment and a list of agents based on the constraints of the environment.
        Note: Sampling without replacement is only restricted to single env candidate.
        This is due to the fact that the number of possible combinations of env and agents is huge.
        Please sample for each env separately if you want to sample without replacement.
        """
        assert (
            not isinstance(agent_classes, list) or len(agent_classes) == n_agent
        ), f"agent_classes should be a list of length {n_agent} or a single agent class"

        if not isinstance(agent_classes, list):
            agent_classes = [agent_classes] * n_agent
        assert (
            len(agents_params) == n_agent
        ), f"agents_params should be a list of length {n_agent}"
        
        assert len(self.filter_func) == n_agent, "Number of filter functions should be equal to number of agents"

        env_profiles: list[EnvironmentProfile] = []
        agents_which_fit_scenario: list[list[str]] = []

        agent_candidate_ids: list[set[str]] | None = None
        if self.agent_candidates:
            agent_candidate_ids = [
                set([cand_id for cand_id in self.agent_candidates if filter_func(cand_id)]) for filter_func in self.filter_func
            ]
            
        else:
            agent_candidate_ids = None
        # print(agent_candidate_ids)
        # print(self.filter_func)
        # print(self.agent_candidates)

        if not replacement:
            assert self.env_candidates and len(self.env_candidates) == 1, (
                "Sampling without replacement is only restricted to single env candidate (must be provided in the constructor). "
                "This is due to the fact that the number of possible combinations of env and agents is huge. "
                "Please sample for each env separately if you want to sample without replacement."
            )

            env_profile_id = (
                self.env_candidates[0].pk
                if not isinstance(self.env_candidates[0], str)
                else self.env_candidates[0]
            )

            assert env_profile_id, "Env candidate must have an id"

            agents_which_fit_scenario = _get_fit_agents_for_one_env(
                env_profile_id, agent_candidate_ids, size
            )
            env_profiles = (
                [EnvironmentProfile.get(env_profile_id)] * size
                if isinstance(self.env_candidates[0], str)
                else [self.env_candidates[0]] * size
            )
        else:
            for _ in range(size):
                if self.env_candidates:
                    env_profile = random.choice(self.env_candidates)
                    if isinstance(env_profile, str):
                        env_profile = EnvironmentProfile.get(env_profile)
                else:
                    env_profile_id = random.choice(list(EnvironmentProfile.all_pks()))
                    env_profile = EnvironmentProfile.get(env_profile_id)
                env_profiles.append(env_profile)
                env_profile_id = env_profile.pk
                assert env_profile_id, "Env candidate must have an id"
                agents_which_fit_scenario.append(
                    _get_fit_agents_for_one_env(env_profile_id, agent_candidate_ids, 1)[
                        0
                    ]
                )

        assert len(env_profiles) == size, "Number of env_profiles is not equal to size"
        assert (
            len(agents_which_fit_scenario) == size
        ), "Number of agents_which_fit_scenario is not equal to size"

        for env_profile, agent_profile_id_list in zip(
            env_profiles, agents_which_fit_scenario
        ):
            env = ParallelSotopiaEnv(env_profile=env_profile, **env_params)
            agent_profiles = [AgentProfile.get(id) for id in agent_profile_id_list]

            agents = [
                agent_class(agent_profile=agent_profile, **agent_params)
                for agent_class, agent_profile, agent_params in zip(
                    agent_classes, agent_profiles, agents_params
                )
            ]
            # set goal for each agent
            for agent, goal in zip(agents, env.profile.agent_goals):
                agent.goal = goal

            yield env, agents