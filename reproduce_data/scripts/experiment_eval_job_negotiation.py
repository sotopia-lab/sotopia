import asyncio
import logging
import os
import sys
import subprocess
from datetime import datetime
from logging import FileHandler
from typing import Any, Generator, cast

import gin
from absl import flags
from rich.logging import RichHandler
from tqdm import tqdm
from typing import Optional, List

from sotopia.agents import LLMAgent
from sotopia.database import (
    AgentProfile,
    EnvAgentComboStorage,
    EnvironmentProfile,
    EpisodeLog,
)
from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    SotopiaDimensions,
    # NegotiationDimensions
)
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.generation_utils.generate import LLM_Name
from sotopia.messages import AgentAction, Observation
from sotopia.samplers import (
    BaseSampler,
    ConstraintBasedSampler,
    EnvAgentCombo,
    FilterBasedSampler,
)
from sotopia.samplers.filter_based_sampler import filter_agent_ids
from sotopia.server import run_async_server
from sotopia_conf.gin_utils import parse_gin_flags, run

_DEFAULT_GIN_SEARCH_PATHS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
]
FLAGS = flags.FLAGS

# date and message only
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

process = subprocess.Popen(
    ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
)
git_head_hash = process.communicate()[0].strip()

logging.basicConfig(
    level=15,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(),
        FileHandler(
            datetime.now().strftime(
                f"./logs/%H_%M_%d_%m_%Y_{str(git_head_hash.decode('utf-8'))}.log"
            )
        ),
    ],
)

env_ids: list[str] = list(EnvironmentProfile.all_pks())
assert all(
    isinstance(env_id, str) for env_id in env_ids
), "env_ids should be a list of strings"


def check_existing_episodes(
    env_id: str,
    agent_ids: list[str],
    models: dict[str, LLM_Name],
    tag: str | None = None,
) -> bool:
    if tag:
        existing_episode = EpisodeLog.find(
            (EpisodeLog.environment == env_id) & (EpisodeLog.tag == tag)
        ).all()
    else:
        existing_episode = EpisodeLog.find(EpisodeLog.environment == env_id).all()
    if existing_episode:
        for episode in existing_episode:
            assert isinstance(episode, EpisodeLog), "episode should be an EpisodeLog"
            if episode.agents == agent_ids and episode.models == list(models.values()):
                return True
        return False
    else:
        return False
big_five_traits = ['Openness to Experience', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
def compose_big_five_target(big_five_target: list[str]) -> str:

    big_five_str = "; ".join([f"{trait} - {target}" for trait, target in zip(big_five_traits, big_five_target)])
    return big_five_str

def _get_agent_ids_by_big_five(big_five_target: Optional[list[str]] = None) -> list[str]:
    agent_candidates: list[AgentProfile] = []
    all_agent_pks = list(AgentProfile.all_pks())
    agent_candidate_id: List[str] = []
    if not big_five_target:
        return all_agent_pks
    
    assert len(big_five_target) == 1 or len(big_five_target) == 5, "big_five_target should be a list of length 1 or 5"
    if len(big_five_target) == 1:
        big_five_target = [big_five_target[0]] * 5

    for agent_pk in all_agent_pks:
        agent_profile = AgentProfile.get(agent_pk)
        if agent_profile.big_five == compose_big_five_target(big_five_target):
            agent_candidate_id.append(agent_pk)
    print(f"In total there are {len(agent_candidate_id)} agents with big five target {big_five_target}")
    return agent_candidate_id

from typing import Callable
def _sample_env_agent_combo_and_push_to_db(env_id: str, agent_candidates: List[str], filters: List[Callable]) -> None:   
    sampler = FilterBasedSampler[Observation, AgentAction](env_candidates=[env_id], agent_candidates=agent_candidates, filter_func=filters)
    env_agent_combo_list = list(
        sampler.sample(agent_classes=[LLMAgent] * 2, replacement=False)
    )
    # print(f"Sampled {len(env_agent_combo_list)} env-agent combos")
    # print(list((agent[0].profile.pk, agent[1].profile.pk) for _, agent in env_agent_combo_list))
    # print([agent.pk for agent in agent_candidates])
    for env, agent in env_agent_combo_list:
        EnvAgentComboStorage(
            env_id=env.profile.pk,
            agent_ids=[agent[0].profile.pk, agent[1].profile.pk],
        ).save()


@gin.configurable
def _iterate_env_agent_combo_not_in_db(
    model_names: dict[str, LLM_Name],
    env_ids: list[str] = [],
    agent_candidate_ids: list[str] = [],
    tag: str | None = None,
    filters: List[Callable] = [],
    batch_size: int = 1,
) -> Generator[EnvAgentCombo[Observation, AgentAction], None, None]:
    """We iterate over each environment and return the **first** env-agent combo that is not in the database."""
    filtered_candidate_ids = filter_agent_ids(filter_funcs=filters, agent_candidate_ids=agent_candidate_ids)
    # print(f"Filtered candidate ids: {[len(candidate) for candidate in filtered_candidate_ids]}")
    
    if not env_ids:
        env_ids = list(EnvironmentProfile.all_pks())
    for env_id in env_ids:
        assert env_id is not None, "env_id should not be None"
        
        for _ in range(batch_size):
            env_agent_combo_storage_list = list(
                EnvAgentComboStorage.find(EnvAgentComboStorage.env_id == env_id).all()
            )
            env_agent_combo_storage_list = [
                combo for combo in env_agent_combo_storage_list if all([combo.agent_ids[idx] in filtered_candidate_ids[idx] for idx in range(len(combo.agent_ids))])
            ]
            
            # env_agent_combo_storage_list = [
            #     combo for combo in env_agent_combo_storage_list if all([agent_id in agent_candidate_ids for agent_id in combo.agent_ids[:1]])
            # ]
            print(f"{len(env_agent_combo_storage_list)} env-agent combos found in the database")
            print(f"w/o filter: {len(list(EnvAgentComboStorage.find(EnvAgentComboStorage.env_id == env_id).all()))}")
            
            
            if not env_agent_combo_storage_list:
                # agent_candidates = [AgentProfile.get(agent_id) for agent_id in agent_candidate_ids]
                _sample_env_agent_combo_and_push_to_db(env_id, agent_candidates=agent_candidate_ids, filters=filters)
                env_agent_combo_storage_list = list(
                    EnvAgentComboStorage.find(EnvAgentComboStorage.env_id == env_id).all()
                )
                env_agent_combo_storage_list = [
                    combo for combo in env_agent_combo_storage_list if all([combo.agent_ids[idx] in filtered_candidate_ids[idx] for idx in range(len(combo.agent_ids))])
                ]
                print("Sampled env_agent_combo: ", len(env_agent_combo_storage_list))
                assert env_agent_combo_storage_list
                
            
            first_env_agent_combo_storage_to_run: EnvAgentComboStorage | None = None
            for env_agent_combo_storage in env_agent_combo_storage_list:
                env_agent_combo_storage = cast(
                    EnvAgentComboStorage, env_agent_combo_storage
                )
                agent_ids = env_agent_combo_storage.agent_ids
                if check_existing_episodes(env_id, agent_ids, model_names, tag):
                    logging.info(
                        f"Episode for {env_id} with agents {agent_ids} using {list(model_names.values())} already exists"
                    )
                    continue
                first_env_agent_combo_storage_to_run = env_agent_combo_storage
                break
            if first_env_agent_combo_storage_to_run:
                env_profile = EnvironmentProfile.get(env_id)
                env = ParallelSotopiaEnv(
                    env_profile=env_profile,
                    model_name=model_names["env"],
                    action_order="round-robin",
                    evaluators=[
                        RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
                    ],
                    terminal_evaluators=[
                        ReachGoalLLMEvaluator(
                            model_names["env"],
                            EvaluationForTwoAgents[SotopiaDimensions],
                        ),
                    ],
                )
                agent_profiles = [AgentProfile.get(id) for id in agent_ids]

                agents = [
                    LLMAgent(agent_profile=agent_profile, model_name=agent_model)
                    for agent_profile, agent_model in zip(
                        agent_profiles,
                        [model_names["agent1"], model_names["agent2"]],
                    )
                ]

                yield env, agents


@gin.configurable
def run_async_server_in_batch(
    *,
    batch_size: int = 1,
    model_names: dict[str, LLM_Name] = {
        "env": "gpt-4",
        "agent1": "gpt-3.5-turbo",
        "agent2": "gpt-3.5-turbo",
    },
    tag: str | None = None,
    verbose: bool = False,
    repeat_time: int = 1,
    agent_ids: list[str] = [],
    env_ids: list[str] = [],
) -> None:
    if not verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        rich_handler = logger.handlers[0]
        logger.removeHandler(rich_handler)
    
    

    # agent_1_filter = lambda agent: AgentProfile.get(agent).occupation == "Hiring Manager"
    agent_1_filter = lambda agent: AgentProfile.get(agent).first_name == "AI"
    agent_2_filter = lambda agent: AgentProfile.get(agent).occupation == "Candidate"
    filters = [agent_1_filter, agent_2_filter]
    print("Total number of envs: ", len(env_ids))
    
    # we cannot get the exact length of the generator, we just give an estimate of the length
    env_agent_combo_iter = _iterate_env_agent_combo_not_in_db(model_names=model_names, env_ids=env_ids, agent_candidate_ids=agent_ids, filters=filters, batch_size=repeat_time)
    env_agent_combo_iter_length = sum(1 for _ in env_agent_combo_iter)

    env_agent_combo_iter = _iterate_env_agent_combo_not_in_db(model_names=model_names, env_ids=env_ids, agent_candidate_ids=agent_ids, filters=filters, batch_size=repeat_time)
    env_agent_combo_batch: list[EnvAgentCombo[Observation, AgentAction]] = []

    while True:
        for env_agent_combo in tqdm(
            env_agent_combo_iter,
            total=env_agent_combo_iter_length,
            desc="Running all envs in batch",
        ): 
            env_agent_combo_batch.append(env_agent_combo)
            if len(env_agent_combo_batch) == batch_size:
                logging.info(
                    f"Running batch of {batch_size} episodes: {env_agent_combo_batch}"
                )
                asyncio.run(
                    run_async_server(
                        model_dict=model_names,
                        sampler=BaseSampler[Observation, AgentAction](),
                        env_agent_combo_list=env_agent_combo_batch,
                        tag=tag,
                    )
                )
                env_agent_combo_batch = []
        else:
            if env_agent_combo_batch:
                logging.info(
                    f"Running batch of {batch_size} episodes: {env_agent_combo_batch}"
                )
                asyncio.run(
                    run_async_server(
                        model_dict=model_names,
                        sampler=BaseSampler[Observation, AgentAction](),
                        env_agent_combo_list=env_agent_combo_batch,
                        tag=tag
                    )
                )
            return


def main(_: Any) -> None:
    parse_gin_flags(
        # User-provided gin paths take precedence if relative paths conflict.
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings,
    )
    
    from sotopia.database.persistent_profile import EnvironmentList
    # env_agent_list = EnvironmentList.find(EnvironmentList.name == "0828_1_hiring").all()
    # envs = env_agent_list[0].environments
    # agents = [index.split("_") for index in env_agent_list[0].agent_index]
    
    target_env_list_name = "hiring"
    target_mode = "competitive"
    
    from sotopia.database.persistent_profile import EnvironmentList
    env_agent_list = EnvironmentList.find(EnvironmentList.name == target_env_list_name).all()
    env_ids = env_agent_list[0].environments
    agent_ids = [index.split("_") for index in env_agent_list[0].agent_index]
    print(env_ids, agent_ids)
    print("In total we have {} envs and {} agent pairs".format(len(env_ids), len(agent_ids)))
    
    for env_id, agent_id in zip(env_ids, agent_ids):
        if target_mode not in EnvironmentProfile.get(env_id).codename:
            raise ValueError(f"Environment {env_id} does not contains {target_mode}")
        
        print(f"Env: {env_id}, Agent: {agent_id}")
        candidate_agent = AgentProfile.get(agent_id[1])
        manager_agent = AgentProfile.get(agent_id[0])
        candidate_agent_bigfive = candidate_agent.personality_and_values.split("Personality Trait: ")[1].split("\n")[0]
        candidate_agent_bigfive = "_".join(candidate_agent_bigfive.split(" "))
        # "you will use a {} method called", help me to extract with regex
        # manager_agent_trust = manager_agent.personality_and_values.split("method called ")[0].split("you will use a")[1].strip()
        # manager_agent_trust = "_".join(manager_agent_trust.split(" "))
        # manager_agent_trust = "manager_trust"
        
        manager_agent_personality = manager_agent.personality_and_values.split("Credibility Persona: ")[1].split("\n")[0]
        attributes = manager_agent_personality.split(", ")
        formatted_attributes = [attr.lower().replace(" ", "_") for attr in attributes]
        # Join the attributes with hyphens
        manager_agent_personality = "-".join(formatted_attributes)
        # python sample_and_upload_to_env.py --name 0923_1_hiring_equal_competitive_bot_transparency_human_bigfive_salary_start_date --environment_file job_scenarios_bot_0922_salary_start_date_equal_competitive.json --agent_file human_agreeableness_ai_transparency.json
        
        suffix = f"trust-bigfive-{manager_agent_personality}-{candidate_agent_bigfive}"
        # suffix = f"{candidate_agent.first_name}{candidate_agent.last_name}"
 
        tag = f"{target_env_list_name}_{suffix}"
        print(f"Running tag {tag}")
        
        MAX_EPISODES = 20
        current_existing_episodes = len(EpisodeLog.find(EpisodeLog.tag == tag).all())
        repeat_time = min(MAX_EPISODES - current_existing_episodes, 10)
        print(f"Current existing episodes: {current_existing_episodes}, repeat time: {repeat_time}")
        
        for i in range(1):
            run_async_server_in_batch(
                agent_ids=agent_id,
                env_ids=[env_id],
                repeat_time=repeat_time,
                tag=tag
            )


if __name__ == "__main__":
    # python sample_and_upload_to_env.py --name 0916_3_hiring_bot_trust_human_bigfive --environment_file job_scenarios_bot.json --agent_file agent_profiles_trust_bigfive.json
    flags.DEFINE_multi_string(
        "gin_file",
        default=None,
        help="Path to gin configuration file. Multiple paths may be passed and "
        "will be imported in the given order, with later configurations  "
        "overriding earlier ones.",
    )

    flags.DEFINE_multi_string(
        "gin_bindings", default=[], help="Individual gin bindings."
    )

    flags.DEFINE_list(
        "gin_search_paths",
        default=["."],
        help="Comma-separated list of gin config path prefixes to be prepended "
        "to suffixes given via `--gin_file`. If a file appears in. Only the "
        "first prefix that produces a valid path for each suffix will be "
        "used.",
    )

    run(main)