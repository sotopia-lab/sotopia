from sotopia.cli.benchmark.benchmark import get_avg_reward
from sotopia.database import EpisodeLog, AgentProfile, EnvironmentProfile
import numpy as np
import json

from unittest.mock import patch

from sotopia.cli.benchmark.benchmark import (
    benchmark,
    benchmark_display,
    run_async_benchmark_in_batch,
)
from unittest import mock
from unittest.mock import create_autospec
from sotopia.cli.benchmark.benchmark import initialize_benchmark_combo
from sotopia.database import EnvAgentComboStorage
import pytest

from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.messages import AgentAction, Observation
from sotopia.samplers import (
    EnvAgentCombo,
)
from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    SotopiaDimensions,
)
from sotopia.agents import LLMAgent

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
    environment = EnvironmentProfile(
        codename="test",
        source="test",
        scenario="Two people are talking",
        agent_goals=[
            "You have 500 dollars and you want to buy the phone",
            "You have a complete new iPhone 16 from Apple Store for $600 and you want to sell it",
        ],
    )

    agent = AgentProfile(first_name="John", last_name="Doe", age=25, pk="John")

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


def get_mock_env_agents_profile() -> tuple[EnvironmentProfile, list[AgentProfile]]:
    env_profile = EnvironmentProfile(
        codename="test",
        source="test",
        scenario="Two people are talking",
        agent_goals=[
            "You have 500 dollars and you want to buy the phone",
            "You have a complete new iPhone 16 from Apple Store for $600 and you want to sell it",
        ],
    )
    agent_1 = AgentProfile(first_name="John", last_name="Doe", age=25, pk="John")
    agent_2 = AgentProfile(first_name="Jane", last_name="Doe", age=25, pk="Jane")

    return env_profile, [agent_1, agent_2]


def mock_success_data() -> list[dict[str, str | list[str]]]:
    data = get_mock_episodes()[:2]
    return [
        {
            "env_id": item.environment,
            "agent_ids": item.agents,
        }
        for item in data
    ]


def mock_success() -> mock.Mock:
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_success_data()
    return mock_response


def mock_fail() -> mock.Mock:
    mock_response = mock.Mock()
    mock_response.status_code = 404
    return mock_response


def compose_env_agent_combo(
    env_profile: EnvironmentProfile, agent_profiles: list[AgentProfile]
) -> EnvAgentCombo[Observation, AgentAction]:
    env = ParallelSotopiaEnv(
        env_profile=env_profile,
        model_name="gpt-4o-mini",
        evaluators=[RuleBasedTerminatedEvaluator(max_turn_number=1, max_stale_turn=2)],
        terminal_evaluators=[
            ReachGoalLLMEvaluator(
                "gpt-4o-mini",
                EvaluationForTwoAgents[SotopiaDimensions],
            )
        ],
    )
    agents = [
        LLMAgent(agent_profile=agent_profile, model_name=agent_model)
        for agent_profile, agent_model in zip(
            agent_profiles, ["gpt-4o-mini", "gpt-4o-mini"]
        )
    ]

    return (env, agents)


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
        extracted_bound, gt_bound, atol=2e-2
    ), f"In error bound, expected {gt_bound}, got {extracted_bound} on dimensions {dimensions}"


mock_delete_function = create_autospec(lambda pk: None)


@patch("requests.get", side_effect=[mock_success(), mock_fail()])
def test_initialize_benchmark_combo(
    mock_requests_get: mock.Mock,
) -> None:
    # Test status code 200
    resp_data = initialize_benchmark_combo(
        url="http://localhost:5000"
    )  # expect to get a non-empty list
    assert (
        isinstance(resp_data, list) and len(resp_data) == 2
    ), f"Expected a length-2 list, but got {resp_data}"
    assert (
        resp_data[0].agent_ids == get_mock_episodes()[0].agents
    ), f"For agents, expected {get_mock_episodes()[0].agents}, but got {resp_data[0].agent_ids}"

    # Test status code 404
    with pytest.raises(ValueError) as excinfo:
        _ = initialize_benchmark_combo(
            url="http://localhost:5000"
        )  # expect to get a ValueError
        assert "Failed to fetch data" in str(excinfo.value)

    # Test get from database
    EnvAgentComboStorage.find = mock.Mock(return_value=EnvAgentComboStorage)  # type: ignore
    EnvAgentComboStorage.all = mock.Mock(return_value=resp_data)  # type: ignore
    new_resp_data = initialize_benchmark_combo(url="")
    assert (
        isinstance(new_resp_data, list) and len(new_resp_data) == 2
    ), f"Expected a length-2 list, but got {new_resp_data}"
    for idx, (resp_item, new_item) in enumerate(zip(resp_data, new_resp_data)):
        assert (
            resp_item.agent_ids == new_item.agent_ids
        ), f"For agents in item {idx}, expected {resp_item.agent_ids}, but got {new_item.agent_ids}"
        assert (
            resp_item.env_id == new_item.env_id
        ), f"For env_id in item {idx}, expected {resp_item.env_id}, but got {new_item.env_id}"


# TODO fix the method assignment issue in mypy, ref: https://github.com/python/mypy/issues/2427
def test_run_async_benchmark_in_batch() -> None:
    return_value = get_mock_episodes()[:10]
    return_value[0].rewards = [0.0, 0.0]

    env, agents = get_mock_env_agents_profile()

    with patch("sotopia.database.EpisodeLog.delete", mock_delete_function), patch(
        "sotopia.database.EpisodeLog.find", return_value=EpisodeLog
    ), patch("sotopia.database.EpisodeLog.all", return_value=return_value), patch(
        "sotopia.database.AgentProfile.find", return_value=AgentProfile
    ), patch("sotopia.database.AgentProfile.all", return_value=agents), patch(
        "sotopia.database.AgentProfile.get",
        return_value=lambda pk: agents[0] if pk == agents[0].pk else agents[1],
    ), patch(
        "sotopia.database.EnvironmentProfile.find", return_value=EnvironmentProfile
    ), patch("sotopia.database.EnvironmentProfile.all", return_value=[env]), patch(
        "sotopia.database.EnvironmentProfile.get", return_value=lambda pk: env
    ):
        assert (
            len(EpisodeLog.find().all()) == 10
        ), f"Expected 10 episodes in the database, but got {len(EpisodeLog.find().all())}"

        env_agent_combo_list = [
            compose_env_agent_combo(
                env_profile=env,
                agent_profiles=agents,
            )
        ]

        run_async_benchmark_in_batch(
            env_agent_combo_list=env_agent_combo_list,
            batch_size=1,
            push_to_db=False,
            tag="test",
        )

        assert (
            mock_delete_function.call_count == 1
        ), f"Expected 1 call to delete, but got {mock_delete_function.call_count}"


mock_initialize = create_autospec(lambda: [])


@patch("sotopia.cli.benchmark.benchmark.initialize_benchmark_combo")
@patch("sotopia.cli.benchmark.benchmark.run_async_benchmark_in_batch")
def test_sotopia_benchmark(
    mock_run_async_benchmark_in_batch: mock.Mock,
    mock_initialize_benchmark_combo: mock.Mock = mock_initialize,
) -> None:
    # Mainly test the benchmark workflow; Assume the benchmark_combo is correct

    with patch("sotopia.database.EpisodeLog.delete", mock_delete_function), patch(
        "sotopia.database.EpisodeLog.find", return_value=EpisodeLog
    ), patch("sotopia.database.EpisodeLog.all", return_value=get_mock_episodes()):
        assert (
            len(EpisodeLog.find().all()) == 20
        ), f"Expected 20 episodes in the database, but got {len(EpisodeLog.find().all())}"

        # `output_to_jsonl` will be tested in the next test, `push_to_db` has been tested elsewhere, so only test `only_show_performance`
        benchmark(
            models=[model_name],
            partner_model="not_test_model",
            evaluator_model="eval_model",
            url="",
            only_show_performance=False,
            output_to_jsonl=False,
            push_to_db=False,
        )
        mock_initialize_benchmark_combo.assert_called_once_with("")

        benchmark(
            models=[model_name],
            partner_model="not_test_model",
            evaluator_model="eval_model",
            url="",
            only_show_performance=True,
            output_to_jsonl=False,
            push_to_db=False,
        )
        mock_initialize_benchmark_combo.assert_called_once_with("")


def test_sotopia_benchmark_display() -> None:
    # Mainly test the average reward calculation (similar to previous get_avg_rewards test)

    with patch("sotopia.database.EpisodeLog.delete", mock_delete_function), patch(
        "sotopia.database.EpisodeLog.find", return_value=EpisodeLog
    ), patch("sotopia.database.EpisodeLog.all", return_value=get_mock_episodes()):
        assert (
            len(EpisodeLog.find().all()) == 20
        ), f"Expected 20 episodes in the database, but got {len(EpisodeLog.find().all())}"
        displayed_stats = benchmark_display(
            model_list=[model_name],
            partner_model="not_test_model",
            evaluator_model="eval_model",
            output_to_jsonl=True,
            save_dir="/tmp",
        )

        target_believability = (7.0, 3.4462887784189147)
        assert np.allclose(
            displayed_stats["test_model"]["believability"],
            target_believability,
            atol=0.02,
        ), f"Got {displayed_stats['test_model']['believability']}, expected {target_believability}"

        benchmark_file = "/tmp/models_vs_not_test_model.jsonl"
        recovered_data = [json.loads(line) for line in open(benchmark_file, "r")]
        target_content = '{"model_name": "test_model", "SOC [-10, 0]": 7.0, "SEC [-10, 0]": 7.0, "FIN [-5, 5]": 7.0, "REL [-5, 5]": 7.0, "KNO [0, 10]": 7.0, "GOAL [0, 10]": 7.0, "BEL [0, 10]": 7.0}'
        assert (
            str(recovered_data[0]).replace("'", '"') == target_content
        ), f"Expected {target_content}, but got {recovered_data[0]}"
