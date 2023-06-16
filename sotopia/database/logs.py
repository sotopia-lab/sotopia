from typing import Any, cast

from pydantic import ConstrainedList, conlist, root_validator
from redis_om import HashModel, JsonModel
from redis_om.model.model import Field  # type: ignore[attr-defined]

from sotopia.database.persistent_profile import AgentProfile
from sotopia.messages import Message


class EpisodeLog(JsonModel):
    # Note that we did not validate the following constraints:
    # 1. The number of turns in messages and rewards should be the same or off by 1
    # 2. The agents in the messages are the same as the agetns

    environment: str = Field(index=True)
    agents: list[str] = Field(index=True)
    messages: list[list[tuple[str, str, str]]]  # Messages arranged by turn
    rewards: list[list[float]]  # Rewards arranged by turn

    @root_validator
    def agent_number_message_number_reward_number_turn_number_match(
        cls, values: Any
    ) -> Any:
        agents, _, rewards = (
            values.get("agents"),
            values.get("messages"),
            values.get("rewards"),
        )
        agent_number = len(agents)
        for rewards_in_turn in rewards:
            assert (
                len(rewards_in_turn) == agent_number
            ), f"Number of agents in rewards {len(rewards_in_turn)} and agents {agent_number} do not match"
        return values

    def render_for_humans(self) -> tuple[list[AgentProfile], list[str]]:
        """Generate a human readable version of the episode log.

        Returns:
            A tuple of (a list of agent_profiles, a list of str): The agent profiles, and the messages and rewards in each turn.
        """

        agent_profiles = [
            AgentProfile.get(pk=uuid_str) for uuid_str in self.agents
        ]
        messages_and_rewards = []
        for idx, turn in enumerate(self.messages):
            messages_and_rewards_in_this_turn = [f"Turn {idx}:"]
            if idx == 0:
                assert (
                    len(turn) >= 2
                ), "The first turn should have at least environemnt messages"
                messages_and_rewards_in_this_turn.append(turn[0][2])
                messages_and_rewards_in_this_turn.append(turn[1][2])
            for sender, receiver, message in turn:
                if receiver == "Environment":
                    if sender != "Environment":
                        messages_and_rewards_in_this_turn.append(
                            f"{sender}: {message}"
                        )
                    else:
                        messages_and_rewards_in_this_turn.append(message)
            if idx < len(self.rewards):
                messages_and_rewards_in_this_turn.append(
                    "The rewards for each agent are :"
                    + " ".join(map(str, self.rewards[idx]))
                )
            messages_and_rewards.append(
                "\n".join(messages_and_rewards_in_this_turn)
            )
        return agent_profiles, messages_and_rewards


class AnnotationForEpisode(JsonModel):
    episode: str = Field(index=True, description="the pk id of episode log")
    annotator_id: str = Field(index=True, full_text_search=True)
    scores_for_each_turn: list[list[int]]
    comments_for_each_turn: list[str] = Field(
        description="optional comments for each turn"
    )
