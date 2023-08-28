import argparse
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, cast

import pandas as pd
import rich
from rich.console import Console
from rich.terminal_theme import MONOKAI

from sotopia.database.logs import EpisodeLog
from sotopia.database.persistent_profile import EnvironmentProfile

os.environ[
    "REDIS_OM_URL"
] = "redis://:AtSeeDFtZvFkYFwSPbZVDMx3@tiger.lti.cs.cmu.edu:6379"


def get_episodes(args: argparse.Namespace) -> list[EpisodeLog]:
    """
    Retrieve episodes with specified tag and models from the database.

    Parameters:
        args: argparse.Namespace containing the tag and model names for filtering.

    Returns:
        List of EpisodeLog objects that match the specified criteria.
    """
    episodes = cast(
        list[EpisodeLog], EpisodeLog.find(EpisodeLog.tag == args.tag).all()
    )

    target_episodes = []
    for episode in episodes:
        if episode.models == [
            args.env_model_name,
            args.agent1_model_name,
            args.agent2_model_name,
        ]:
            target_episodes.append(episode)

    print("Total number of episodes: ", len(target_episodes))
    return target_episodes


def filter_episodes_and_change_tag(
    args: argparse.Namespace, episodes: list[EpisodeLog]
) -> list[EpisodeLog]:
    """
    Filter episodes by environment and change their tag.

    Parameters:
        args: argparse.Namespace containing filtering criteria and new tag.
        episodes: List of EpisodeLog objects to be filtered.

    Returns:
        List of filtered EpisodeLog objects with updated tag.
    """
    env_set = set()
    episode_env_dict: dict[str, list[EpisodeLog]] = {}

    for episode in episodes:
        env = episode.environment
        if env not in env_set:
            env_set.add(env)
            episode_env_dict[env] = []
        episode_env_dict[env].append(episode)

    print("Total number of environments: ", len(env_set))

    filtered_episodes = []
    for env, episodes in episode_env_dict.items():
        correct_episodes = []
        for idx, episode in enumerate(episodes):
            if episodes[idx].models == [
                args.env_model_name,
                args.agent1_model_name,
                args.agent2_model_name,
            ]:
                correct_episodes.append(episodes[idx])

        filtered_episodes += correct_episodes[: args.each_env_episode_num]

    print("Total number of filtered episodes: ", len(filtered_episodes))
    return filtered_episodes


def update_episodes_in_redis(
    args: argparse.Namespace, episodes: list[EpisodeLog]
) -> None:
    """
    Save the list of episodes to Redis.

    Parameters:
        episodes: List of EpisodeLog objects to be saved.
    """
    assert len(episodes) == 450
    for episode in episodes:
        episode.update(tag=args.target_tag)  # type: ignore [attr-defined]


def export_html_from_log(
    args: argparse.Namespace, episodes: list[EpisodeLog]
) -> None:
    """
    Export episodes as HTML files for each unique environment codename.

    Parameters:
        args: argparse.Namespace containing output directory information.
        episodes: List of EpisodeLog objects to be exported.
    """
    output_directory = args.output_directory or "./"

    codename_set = set()
    for episode in episodes:
        assert episode.models is not None, "Episode models should not be None"

        model1 = episode.models[1].replace(".", "")
        model2 = episode.models[2].replace(".", "")

        codename = (
            EnvironmentProfile()
            .get(episode.environment)
            .codename.replace(" ", "_")
        )

        if codename not in codename_set:
            codename_set.add(codename)
            file_path = os.path.join(
                output_directory,
                f"{model1}-{model2}-{codename}.html".replace("/", "_"),
            )

            agent_profiles, conversation = episode.render_for_humans()

            console = Console(record=True, log_time=False, log_path=False)
            console.log(
                f"Models:\n Env: {episode.models[0]}\n Agent1: {episode.models[1]}\n Agent2: {episode.models[2]}\n"
            )
            for agent_profile in agent_profiles:
                console.log(agent_profile)
            for message in conversation:
                console.log(message)

            console.save_svg(file_path, theme=MONOKAI)
        else:
            continue


def render_for_human(episodes: list[EpisodeLog]) -> None:
    """
    Render a human-readable version of an episode.

    Parameters:
        episodes: List of EpisodeLog objects, where the 2nd one will be rendered for human readability.
    """
    # get a human readable version of the episode
    agent_profiles, conversation = episodes[1].render_for_humans()
    for agent_profile in agent_profiles:
        rich.print(agent_profile)
    for message in conversation:
        rich.print(message)


def get_avg_reward_for_models(
    args: argparse.Namespace, episodes: list[EpisodeLog]
) -> pd.DataFrame:
    """Get the average reward for each model in the episodes.

    Args:
        episodes (list[EpisodeLog]): A list of episodes.

    Returns:
        dict[str, float]: A dictionary mapping model names to average rewards.
    """
    model_rewards: Dict[
        str, List[Union[Tuple[float, Dict[str, float]], Any]]
    ] = defaultdict(list)
    model_rewards_avg: dict[str, dict[str, float]] = {}
    for episode in episodes:
        assert episode.models is not None, "Episode models should not be None"
        for idx, model in enumerate(episode.models[1:]):  # skip env
            if isinstance(episode.rewards[idx], tuple):
                if args.verbose:
                    print(episode.rewards[idx])
                    print(model)
                    print("=" * 50)
                model_rewards[model + "_agent_" + str(idx)].append(
                    episode.rewards[idx]
                )
            else:
                if args.verbose:
                    print(episode.rewards)
    for model in model_rewards:
        model_rewards[model] = [rewards[1] for rewards in model_rewards[model]]
        df = pd.DataFrame(model_rewards[model])
        avg_dict = df.mean(axis=0).to_dict()
        model_rewards_avg[model] = avg_dict
    return pd.DataFrame.from_dict(model_rewards_avg)


def get_avg_successRate_for_models(
    args: argparse.Namespace, episodes: list[EpisodeLog]
) -> pd.DataFrame:
    """Get the average success rate for each model in the episodes.

    Args:
        episodes (list[EpisodeLog]): A list of episodes.

    Returns:
        dict[str, float]: A dictionary mapping model names to average success rates.
    """
    model_rewards: Dict[
        str, List[Union[Tuple[float, Dict[str, float]], Any]]
    ] = defaultdict(list)
    model_successRate_avg: dict[str, pd.DataFrame] = {}
    for episode in episodes:
        assert episode.models is not None, "Episode models should not be None"
        for idx, model in enumerate(episode.models[1:]):  # skip env
            if isinstance(episode.rewards[idx], tuple):
                model_rewards[model + "_agent_" + str(idx)].append(
                    episode.rewards[idx]
                )
            else:
                if args.verbose:
                    print(episode.rewards, episode.messages[0])

    for model in model_rewards:
        model_successRate_avg[model] = pd.DataFrame(
            [rewards[1] for rewards in model_rewards[model]]
        )

    assert len(model_successRate_avg) == 2, "There should be two models"
    model_list = list(model_successRate_avg.keys())
    model_one_successRate = (
        model_successRate_avg[model_list[0]]
        > model_successRate_avg[model_list[1]]
    )
    model_two_successRate = (
        model_successRate_avg[model_list[0]]
        < model_successRate_avg[model_list[1]]
    )
    model_on_par_successRate = (
        model_successRate_avg[model_list[0]]
        == model_successRate_avg[model_list[1]]
    )
    return pd.DataFrame.from_dict(
        {
            model_list[0]: model_one_successRate.mean(axis=0).to_dict(),
            "on_par": model_on_par_successRate.mean(axis=0).to_dict(),
            model_list[1]: model_two_successRate.mean(axis=0).to_dict(),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag", type=str, help="tag of the experiment", required=True
    )
    parser.add_argument(
        "--target_tag", type=str, help="tag of the experiment", default=None
    )
    parser.add_argument(
        "--agent1_model_name",
        type=str,
        choices=[
            "gpt-3.5-turbo",
            "gpt-4",
            "togethercomputer/llama-2-70b-chat",
        ],
        required=True,
    )
    parser.add_argument(
        "--agent2_model_name",
        type=str,
        choices=[
            "gpt-3.5-turbo",
            "gpt-4",
            "togethercomputer/llama-2-70b-chat",
        ],
        required=True,
    )
    parser.add_argument(
        "--output_directory", type=str, help="output directory", default="./"
    )
    parser.add_argument(
        "--env_model_name", type=str, choices=["gpt-4"], default="gpt-4"
    )
    parser.add_argument("--export_html", action="store_true")

    parser.add_argument("--do_filter", action="store_true")
    parser.add_argument("--do_update", action="store_true")
    parser.add_argument("--each_env_episode_num", type=int, default=5)
    parser.add_argument("--env_num", type=int, default=90)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    episodes = get_episodes(args)

    if args.export_html:
        export_html_from_log(args, episodes)
    if args.do_filter:
        episodes = filter_episodes_and_change_tag(args, episodes)
    if args.do_update:
        update_episodes_in_redis(args, episodes)

    assert args.env_num * args.each_env_episode_num == len(
        episodes
    ), "The number of episodes is not correct"
    avg_reward = get_avg_reward_for_models(args, episodes)
    avg_successRate = get_avg_successRate_for_models(args, episodes)
    print(avg_reward)
    print(avg_successRate)
