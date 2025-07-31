import asyncio
import logging
import os
import subprocess
from datetime import datetime
from logging import FileHandler
from typing import Any, Generator, cast, Optional, List, Callable

import gin
from absl import flags
from rich.logging import RichHandler
from tqdm import tqdm
# Added for transparency-aware agents
from sotopia.transparency_hook import make_transparency_agent
from sotopia.agents import LLMAgent  # still used for sampling helper
from sotopia.database import (
    AgentProfile,
    EnvAgentComboStorage,
    EnvironmentProfile,
    EpisodeLog,
)
from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    EpisodeLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    SotopiaDimensions,
)
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.messages import AgentAction, Observation
from sotopia.samplers import (
    BaseSampler,
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
    models: dict[str, str],
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
            assert isinstance(episode, EpisodeLog)
            if episode.agents == agent_ids and episode.models == list(models.values()):
                return True
        return False
    else:
        return False

# ---------------------------------------------------------------------------
# Helpers for Big-5 filtering (kept from original script)
# ---------------------------------------------------------------------------

big_five_traits = [
    'Openness to Experience',
    'Conscientiousness',
    'Extraversion',
    'Agreeableness',
    'Neuroticism',
]

def compose_big_five_target(big_five_target: list[str]) -> str:
    return "; ".join(
        [f"{trait} - {target}" for trait, target in zip(big_five_traits, big_five_target)]
    )


def _get_agent_ids_by_big_five(big_five_target: Optional[list[str]] = None) -> list[str]:
    all_agent_pks = list(AgentProfile.all_pks())
    if not big_five_target:
        return all_agent_pks

    assert len(big_five_target) in (1, 5), "big_five_target should be length 1 or 5"
    if len(big_five_target) == 1:
        big_five_target = big_five_target * 5  # type: ignore[misc]

    agent_candidate_id: List[str] = []
    for agent_pk in all_agent_pks:
        agent_profile = AgentProfile.get(agent_pk)
        if agent_profile.big_five == compose_big_five_target(big_five_target):
            agent_candidate_id.append(agent_pk)
    logging.info(
        f"In total there are {len(agent_candidate_id)} agents with big five target {big_five_target}"
    )
    return agent_candidate_id

# ---------------------------------------------------------------------------
# Sampling helper – OK to use plain LLMAgent here because we only need pks
# ---------------------------------------------------------------------------

def _sample_env_agent_combo_and_push_to_db(
    env_id: str,
    agent_candidates: List[str],
    filters: List[Callable],
) -> None:
    sampler = FilterBasedSampler[Observation, AgentAction](
        env_candidates=[env_id],
        agent_candidates=agent_candidates,
        filter_func=filters,
    )
    env_agent_combo_list = list(
        sampler.sample(agent_classes=[LLMAgent] * 2, replacement=False)
    )
    for env, agent in env_agent_combo_list:
        EnvAgentComboStorage(
            env_id=env.profile.pk,
            agent_ids=[agent[0].profile.pk, agent[1].profile.pk],
        ).save()

# ---------------------------------------------------------------------------
# Core iterator that instantiates transparency-aware agents
# ---------------------------------------------------------------------------

@gin.configurable
def _iterate_env_agent_combo_not_in_db(
    model_names: dict[str, str],
    env_ids: list[str] = [],
    agent_candidate_ids: list[str] = [],
    tag: str | None = None,
    filters: List[Callable] = [],
    batch_size: int = 1,
) -> Generator[EnvAgentCombo[Observation, AgentAction], None, None]:
    """Yield env-agent combos that haven’t been used yet."""

    filtered_candidate_ids = filter_agent_ids(
        filter_funcs=filters, agent_candidate_ids=agent_candidate_ids
    )
    logging.info(
        f"Filtered candidate ids: {[len(candidate) for candidate in filtered_candidate_ids]}"
    )

    if not env_ids:
        env_ids = list(EnvironmentProfile.all_pks())

    for env_id in env_ids:
        assert env_id is not None, "env_id should not be None"

        for _ in range(batch_size):
            env_agent_combo_storage_list = list(
                EnvAgentComboStorage.find(
                    EnvAgentComboStorage.env_id == env_id
                ).all()
            )
            env_agent_combo_storage_list = [
                combo
                for combo in env_agent_combo_storage_list
                if all(
                    [
                        combo.agent_ids[idx] in filtered_candidate_ids[idx]
                        for idx in range(len(combo.agent_ids))
                    ]
                )
            ]

            if not env_agent_combo_storage_list:
                _sample_env_agent_combo_and_push_to_db(
                    env_id, agent_candidates=agent_candidate_ids, filters=filters
                )
                env_agent_combo_storage_list = list(
                    EnvAgentComboStorage.find(
                        EnvAgentComboStorage.env_id == env_id
                    ).all()
                )
                env_agent_combo_storage_list = [
                    combo
                    for combo in env_agent_combo_storage_list
                    if all(
                        [
                            combo.agent_ids[idx] in filtered_candidate_ids[idx]
                            for idx in range(len(combo.agent_ids))
                        ]
                    )
                ]
                logging.info(
                    "Sampled env_agent_combo:", len(env_agent_combo_storage_list)
                )
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
                        EpisodeLLMEvaluator(
                            model_names["env"],
                            EvaluationForTwoAgents[SotopiaDimensions],
                        ),
                    ],
                )
                agent_profiles = [AgentProfile.get(id) for id in agent_ids]

                # Create agents with transparency control
                agents = [
                    make_transparency_agent(agent_profile, agent_model, tag)
                    for agent_profile, agent_model in zip(
                        agent_profiles,
                        [model_names["agent1"], model_names["agent2"]],
                    )
                ]

                yield env, agents

# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

@gin.configurable
def run_async_server_in_batch(
    *,
    batch_size: int = 10,
    model_names: dict[str, str] = {
        "env": "gpt-4",
        "agent1": "gpt-4o",
        "agent2": "gpt-4o",
    },
    tag: str | None = None,
    verbose: bool = False,
    repeat_time: int = 10,
    agent_ids: list[str] = [],
    env_ids: list[str] = [],
) -> None:
    if not verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        rich_handler = logger.handlers[0]
        logger.removeHandler(rich_handler)

    agent_1_filter = lambda agent: AgentProfile.get(agent).first_name == "AI"
    agent_2_filter = lambda agent: AgentProfile.get(agent).occupation == "Candidate"
    filters = [agent_1_filter, agent_2_filter]

    logging.info("Total number of envs: %d", len(env_ids))

    # First iterator to compute length
    env_agent_combo_iter = _iterate_env_agent_combo_not_in_db(
        model_names=model_names,
        env_ids=env_ids,
        agent_candidate_ids=agent_ids,
        filters=filters,
        batch_size=repeat_time,
        tag=tag,  # << pass tag
    )
    env_agent_combo_iter_length = sum(1 for _ in env_agent_combo_iter)

    # Second iterator to actually iterate
    env_agent_combo_iter = _iterate_env_agent_combo_not_in_db(
        model_names=model_names,
        env_ids=env_ids,
        agent_candidate_ids=agent_ids,
        filters=filters,
        batch_size=repeat_time,
        tag=tag,  # << pass tag
    )

    env_agent_combo_batch: list[EnvAgentCombo[Observation, AgentAction]] = []

    while True:
        for env_agent_combo in tqdm(
            env_agent_combo_iter,
            total=env_agent_combo_iter_length,
            desc="Running all envs in batch",
        ):
            env_agent_combo_batch.append(env_agent_combo)
            logging.info("Length of env_agent_combo_batch", len(env_agent_combo_batch))
            if len(env_agent_combo_batch) == batch_size:
                logging.info("Running batch of %d episodes", batch_size)
                asyncio.run(
                    run_async_server(
                        model_dict=model_names,
                        sampler=BaseSampler[Observation, AgentAction](),
                        env_agent_combo_list=env_agent_combo_batch,
                        tag=tag,
                        push_to_db=True,
                    )
                )
                env_agent_combo_batch = []
        else:
            if env_agent_combo_batch:
                logging.info("Running final batch of %d episodes", len(env_agent_combo_batch))
                asyncio.run(
                    run_async_server(
                        model_dict=model_names,
                        sampler=BaseSampler[Observation, AgentAction](),
                        env_agent_combo_list=env_agent_combo_batch,
                        tag=tag,
                        push_to_db=True,
                    )
                )
            return

# ---------------------------------------------------------------------------
# Main entry point – mostly unchanged except for imports
# ---------------------------------------------------------------------------

def main(_: Any) -> None:
    parse_gin_flags(
        FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
        FLAGS.gin_file,
        FLAGS.gin_bindings,
    )

    from sotopia.database.persistent_profile import EnvironmentList

    # target_env_list_name = "sotopia_transparency_experiments_job_hiring_competitive"
    # target_mode = "competitive"
    target_env_list_name = "sotopia_transparency_experiments_job_hiring_cooperative"
    target_mode = "cooperative"

    env_agent_list = EnvironmentList.find(EnvironmentList.name == target_env_list_name).all()
    env_ids = env_agent_list[0].environments
    agent_ids = [index.split("_") for index in env_agent_list[0].agent_index]

    logging.info("%s envs, %s agent pairs", len(env_ids), len(agent_ids))

    for i, (env_id, agent_id) in enumerate(zip(env_ids, agent_ids), start=1):
        if target_mode not in EnvironmentProfile.get(env_id).codename:
            raise ValueError(f"Environment {env_id} does not contain {target_mode}")

        candidate_agent = AgentProfile.get(agent_id[1])
        manager_agent = AgentProfile.get(agent_id[0])

        candidate_trait = (
            candidate_agent.personality_and_values.split("Personality Trait: ")[1].split("\n")[0]
        )
        candidate_trait = "_".join(candidate_trait.split())

        if "Credibility Persona: " in manager_agent.personality_and_values:
            persona_line = manager_agent.personality_and_values.split("Credibility Persona: ")[1].split("\n")[0]
            manager_persona = "-".join(
                attr.strip().replace(" ", "_").lower() for attr in persona_line.split(", ")
            )
        else:
            manager_persona = "unknown"

        suffix = f"trust1-bigfive-{manager_persona}-{candidate_trait}"
        tag = f"{target_env_list_name}_{suffix}_{i}"
        print("tag", tag)
        logging.info("Running tag %s", tag)

        MAX_EPISODES = 20
        existing = len(EpisodeLog.find(EpisodeLog.tag == tag).all())
        repeat_time = 10
        if repeat_time == 0:
            logging.info("All %d episodes already exist for tag %s", MAX_EPISODES, tag)
            continue

        run_async_server_in_batch(
            agent_ids=agent_id,
            env_ids=[env_id],
            repeat_time=repeat_time,
            tag=tag,
        )


if __name__ == "__main__":
    flags.DEFINE_multi_string(
        "gin_file",
        default=None,
        help="Path to gin configuration file.",
    )

    flags.DEFINE_multi_string(
        "gin_bindings", default=[], help="Individual gin bindings."
    )

    flags.DEFINE_list(
        "gin_search_paths",
        default=["."],
        help="Comma-separated list of gin config path prefixes to be prepended to suffixes given via --gin_file.",
    )

    run(main) 