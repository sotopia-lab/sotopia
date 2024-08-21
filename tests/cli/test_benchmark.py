import pytest
from sotopia.cli.benchmark.benchmark import get_avg_reward
from sotopia.database import EpisodeLog, AgentProfile, EnvironmentProfile
import numpy as np

def test_get_rewards() -> None:
    # all_episodes = EpisodeLog.find().all()
    environment = EnvironmentProfile.find().all()[0]
    agent = AgentProfile.find().all()[0]
    
    model_name = "test_model"
    model_pairs = [["eval_model", "test_model", "not_test_model"], ["eval_model", "not_test_model", "test_model"]]
    
    dimensions = ["believability", "relationship", "knowledge", "secret", "social_rules", "financial_and_material_benefits", "goal"] 
    agent1_rewards = [float(i) for i, _ in enumerate(dimensions)] 
    agent2_rewards = [14. - reward for reward in agent1_rewards]
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
    
    target_rewards = [[base_agent1_rewards, base_agent2_rewards], [base_agent2_rewards, base_agent1_rewards]]
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
            
    test_rewards = get_avg_reward(target_episodes, model_name)
    print(test_rewards)
    # 1. check the episode count
    assert test_rewards["episode_count"][0] == len(target_episodes), f"Expected {len(target_episodes)}, got {test_rewards['episode_count']}"
    
    # 2. check average rewards
    extracted_rewards = [test_rewards[dim][0] for dim in dimensions]
    gt_rewards = [7. for _ in dimensions]
    assert extracted_rewards == gt_rewards, f"In average, expected {gt_rewards}, got {extracted_rewards} on dimensions {dimensions}"
    
    # 3. check the error bound
    extracted_bound = [test_rewards[dim][1] for dim in dimensions]
    sem = [1.6499, 1.4142, 1.1785, 0.9428, 0.7071, 0.4714, 0.2357]
    # for [0, 1, 2, 3, 4, 5, 6] * 5 + [14, 13, 12, 11, 10, 9, 8] * 5
    # ppf = 2.093 for 95% confidence interval with 20 samples
    gt_bound = [sem[i] * 2.093 for i in range(len(dimensions))]
    assert np.allclose(extracted_bound, gt_bound, atol=1e-2), f"In error bound, expected {gt_bound}, got {extracted_bound} on dimensions {dimensions}"
     