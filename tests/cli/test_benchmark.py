from sotopia.cli.benchmark.benchmark import get_avg_reward
from sotopia.database import EpisodeLog, AgentProfile, EnvironmentProfile
import numpy as np
from unittest.mock import patch

from sotopia.cli.benchmark.benchmark import (
    benchmark,
    benchmark_display,
    run_async_benchmark_in_batch,
)
from unittest import mock
from unittest.mock import create_autospec

dimensions = [
    "believability",
    "relationship",
    "knowledge",
    "secret",
    "social_rules",
    "financial_and_material_benefits",
    "goal",
]
model_pairs = [
    ["eval_model", "test_model", "not_test_model"],
    ["eval_model", "not_test_model", "test_model"],
]
model_name = "test_model"


def get_mock_episodes() -> list[EpisodeLog]:
    # all_episodes = EpisodeLog.find().all()
    environment = EnvironmentProfile.find().all()[0]
    agent = AgentProfile.find().all()[0]

    agent1_rewards = [float(i) for i, _ in enumerate(dimensions)]
    agent2_rewards = [14.0 - reward for reward in agent1_rewards]
    # agent2_rewards: reversed(agent1_rewards)
    base_agent1_rewards = (
        sum(agent1_rewards) / len(agent1_rewards),
        {dim: reward for dim, reward in zip(dimensions, agent1_rewards)},
    )
    base_agent2_rewards = (
        sum(agent2_rewards) / len(agent2_rewards),
        {dim: reward for dim, reward in zip(dimensions, agent2_rewards)},
    )
    TEST_EPISODE_NUM = 5

    target_rewards = [
        [base_agent1_rewards, base_agent2_rewards],
        [base_agent2_rewards, base_agent1_rewards],
    ]
    target_episodes = []

    for agent_pos in range(2):
        for reward_pos in range(2):
            for _ in range(TEST_EPISODE_NUM):
                episode = EpisodeLog(
                    environment=environment.pk,
                    agents=[agent.pk, agent.pk],
                    tag="test",
                    models=model_pairs[agent_pos],
                    messages=[],
                    reasoning="",
                    rewards=target_rewards[reward_pos],
                    rewards_prompt="",
                )
                target_episodes.append(episode)
    return target_episodes


def test_get_rewards() -> None:
    target_episodes = get_mock_episodes()

    test_rewards = get_avg_reward(target_episodes, model_name)

    # 1. check the episode count
    assert test_rewards["episode_count"][0] == len(
        target_episodes
    ), f"Expected {len(target_episodes)}, got {test_rewards['episode_count']}"

    # 2. check average rewards
    extracted_rewards = [test_rewards[dim][0] for dim in dimensions]
    gt_rewards = [7.0 for _ in dimensions]
    assert (
        extracted_rewards == gt_rewards
    ), f"In average, expected {gt_rewards}, got {extracted_rewards} on dimensions {dimensions}"

    # 3. check the error bound
    extracted_bound = [test_rewards[dim][1] for dim in dimensions]
    sem = [1.6499, 1.4142, 1.1785, 0.9428, 0.7071, 0.4714, 0.2357]
    # for [0, 1, 2, 3, 4, 5, 6] * 5 + [14, 13, 12, 11, 10, 9, 8] * 5
    # ppf = 2.093 for 95% confidence interval with 20 samples
    gt_bound = [sem[i] * 2.093 for i in range(len(dimensions))]
    assert np.allclose(
        extracted_bound, gt_bound, atol=1e-2
    ), f"In error bound, expected {gt_bound}, got {extracted_bound} on dimensions {dimensions}"


mock_delete_function = create_autospec(lambda pk: None)


# TODO fix the method assignment issue in mypy, ref: https://github.com/python/mypy/issues/2427
@patch("sotopia.server.run_async_server")
def test_run_async_benchmark_in_batch(
    mock_run_async_server: mock.Mock,
) -> None:
    # Mainly test the deletion; Assume the `run_async_server` is correct;
    return_value = get_mock_episodes()[:10]
    return_value[0].rewards = [0.0, 0.0]

    EpisodeLog.delete = mock_delete_function  # type: ignore
    EpisodeLog.find = mock.Mock(return_value=EpisodeLog)  # type: ignore
    EpisodeLog.all = mock.Mock(return_value=return_value)  # type: ignore

    assert (
        len(EpisodeLog.find().all()) == 10
    ), f"Expected 10 episodes in the database, but got {len(EpisodeLog.find().all())}"

    run_async_benchmark_in_batch(
        env_agent_combo_list=[],
    )

    assert (
        mock_delete_function.call_count == 1
    ), f"Expected 1 call to delete, but got {mock_delete_function.call_count}"


@patch("sotopia.cli.benchmark.benchmark.run_async_benchmark_in_batch")
@patch("sotopia.cli.benchmark.benchmark.initialize_benchmark_combo")
def test_sotopia_benchmark(
    mock_initialize_benchmark_combo: mock.Mock,
    mock_run_async_benchmark_in_batch: mock.Mock,
) -> None:
    # Mainly test the benchmark workflow; Assume the benchmark_combo is correct
    EpisodeLog.find = mock.Mock(return_value=EpisodeLog)  # type: ignore
    EpisodeLog.all = mock.Mock(return_value=get_mock_episodes())  # type: ignore
    EpisodeLog.delete = mock_delete_function  # type: ignore

    assert (
        len(EpisodeLog.find().all()) == 20
    ), f"Expected 20 episodes in the database, but got {len(EpisodeLog.find().all())}"
    mock_initialize_benchmark_combo.return_value = []

    benchmark(
        models=[model_name],
        partner_model="not_test_model",
        evaluator_model="eval_model",
        only_show_performance=False,
        url="",
    )


def test_sotopia_benchmark_display() -> None:
    # Mainly test the average reward calculation (similar to previous get_avg_rewards test)
    EpisodeLog.find = mock.Mock(return_value=EpisodeLog)  # type: ignore
    EpisodeLog.all = mock.Mock(return_value=get_mock_episodes())  # type: ignore

    assert (
        len(EpisodeLog.find().all()) == 20
    ), f"Expected 20 episodes in the database, but got {len(EpisodeLog.find().all())}"
    displayed_stats = benchmark_display(
        model_list=[model_name],
        partner_model="not_test_model",
        evaluator_model="eval_model",
    )

    target_believability = (7.0, 3.4462887784189147)
    assert np.allclose(
        displayed_stats["test_model"]["believability"], target_believability, atol=0.02
    ), f"Got {displayed_stats['test_model']['believability']}, expected {target_believability}"
