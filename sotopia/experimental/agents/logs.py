import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from pydantic import model_validator, BaseModel
from redis_om import JsonModel
from redis_om.model.model import Field
from typing import Literal, Dict, List, Any, Optional
from sotopia.database.persistent_profile import AgentProfile
import json


class MessageContext(BaseModel):
    """Model for message context information"""

    content: str
    target_agents: List[str] = []
    target_groups: List[str] = []
    context: Literal["group", "individual", "broadcast", "response"]
    responding_to: Optional[Dict[str, Any]] = None


class NonStreamingSimulationStatus(JsonModel):
    episode_pk: str = Field(index=True)
    status: Literal["Started", "Error", "Completed"]


class BaseEpisodeLog(BaseModel):
    # Note that we did not validate the following constraints:
    # 1. The number of turns in messages and rewards should be the same or off by 1
    # 2. The agents in the messages are the same as the agents
    environment: str = Field(index=True)
    agents: list[str] = Field(index=True)
    tag: str | None = Field(index=True, default="")
    models: list[str] | None = Field(index=True, default=[])
    messages: list[
        list[tuple[str, str, str]]
    ]  # Messages arranged by turn: (sender, receiver, content)
    reasoning: str = Field(default="")
    rewards: list[tuple[float, dict[str, float]] | float]  # Rewards arranged by turn
    rewards_prompt: str

    # Added field for group messaging support
    groups: Dict[str, List[str]] = Field(
        default_factory=dict
    )  # Group name -> list of agent names

    @model_validator(mode="after")
    def agent_number_message_number_reward_number_turn_number_match(self) -> Self:
        agent_number = len(self.agents)
        assert (
            len(self.rewards) == agent_number
        ), f"Number of agents in rewards {len(self.rewards)} and agents {agent_number} do not match"
        return self

    def render_for_humans(self) -> tuple[list[AgentProfile], list[str]]:
        """Generate a human readable version of the episode log.
        Returns:
            A tuple of (a list of agent_profiles, a list of str): The agent profiles, and the messages and rewards in each turn.
        """
        agent_profiles = [AgentProfile.get(pk=uuid_str) for uuid_str in self.agents]
        messages_and_rewards = []
        for idx, turn in enumerate(self.messages):
            messages_in_this_turn = []
            if idx == 0:
                assert (
                    len(turn) >= 2
                ), "The first turn should have at least environment messages"
                messages_in_this_turn.append(turn[0][2])
                messages_in_this_turn.append(turn[1][2])
            for sender, receiver, message in turn:
                # Try to parse as JSON context
                try:
                    message_data = json.loads(message)
                    if isinstance(message_data, dict) and "content" in message_data:
                        # This is a formatted message with context
                        content = message_data["content"]
                        context = message_data.get("context", "")

                        if context == "group":
                            # Format group messages
                            group_name = receiver.replace("Group:", "")
                            messages_in_this_turn.append(
                                f"{sender} to group {group_name}: {content}"
                            )
                        elif context == "individual":
                            # Format individual messages
                            target = receiver.replace("Agent:", "")
                            messages_in_this_turn.append(
                                f"{sender} to {target}: {content}"
                            )
                        elif context == "response":
                            # Format responses
                            response_to = receiver.replace("Response:", "")
                            messages_in_this_turn.append(
                                f"{sender} responds to {response_to}: {content}"
                            )
                        elif context == "broadcast":
                            # Format broadcast messages
                            messages_in_this_turn.append(
                                f"{sender} (broadcast): {content}"
                            )
                        else:
                            # Default format for unknown context
                            messages_in_this_turn.append(
                                f"{sender} to {receiver}: {content}"
                            )
                    else:
                        # Default handling
                        messages_in_this_turn.append(
                            f"{sender} to {receiver}: {message}"
                        )
                except (json.JSONDecodeError, TypeError):
                    # Handle legacy message format
                    if receiver == "Environment":
                        if sender != "Environment":
                            if "did nothing" in message:
                                continue
                            else:
                                if "said:" in message:
                                    messages_in_this_turn.append(f"{sender} {message}")
                                else:
                                    messages_in_this_turn.append(f"{sender}: {message}")
                        else:
                            messages_in_this_turn.append(message)
                    else:
                        messages_in_this_turn.append(
                            f"{sender} to {receiver}: {message}"
                        )

            messages_and_rewards.append("\n".join(messages_in_this_turn))

        messages_and_rewards.append(f"The reasoning is:\n{self.reasoning}")

        # Format rewards
        reward_lines = ["The rewards are:"]
        for i, reward in enumerate(self.rewards):
            reward_lines.append(f"Agent {i+1}: {reward}")
        messages_and_rewards.append("\n".join(reward_lines))

        return agent_profiles, messages_and_rewards

    def get_message_for_client(self) -> Dict[str, Any]:
        """
        Convert the episode log to a client-friendly format
        that includes properly formatted messages with context.
        """
        formatted_messages = []

        for turn in self.messages:
            turn_messages = []
            for sender, receiver, message in turn:
                try:
                    # Try to parse as JSON
                    msg_data = json.loads(message)
                    if isinstance(msg_data, dict) and "content" in msg_data:
                        # Create a clean client message
                        client_msg = {
                            "sender": sender,
                            "receiver": receiver,
                            "content": msg_data["content"],
                            "context": msg_data.get("context", ""),
                            "target_agents": msg_data.get("target_agents", []),
                            "target_groups": msg_data.get("target_groups", []),
                        }

                        # Add response context if available
                        if "responding_to" in msg_data:
                            client_msg["responding_to"] = msg_data["responding_to"]

                        turn_messages.append(client_msg)
                    else:
                        # Simple message without context
                        turn_messages.append(
                            {
                                "sender": sender,
                                "receiver": receiver,
                                "content": message,
                                "context": "regular",
                            }
                        )
                except (json.JSONDecodeError, TypeError):
                    # Legacy message format
                    turn_messages.append(
                        {
                            "sender": sender,
                            "receiver": receiver,
                            "content": message,
                            "context": "regular",
                        }
                    )

            if turn_messages:
                formatted_messages.append(turn_messages)

        return {
            "environment": self.environment,
            "agents": self.agents,
            "messages": formatted_messages,
            "groups": self.groups,
        }


class EpisodeLog(BaseEpisodeLog, JsonModel):
    """Redis-compatible episode log with enhanced message context support"""
    pass


class AnnotationForEpisode(JsonModel):
    episode: str = Field(index=True, description="the pk id of episode log")
    annotator_id: str = Field(index=True, full_text_search=True)
    rewards: list[tuple[float, dict[str, float]] | float]
    reasoning: str
