from typing import TypedDict

from sotopia.agents import Agents, LLMAgent
from sotopia.database import AgentProfile, EpisodeLog, EnvironmentProfile
from sotopia.envs.parallel import (
    ParallelSotopiaEnv,
    render_text_for_agent,
    render_text_for_environment,
)
from sotopia.messages import Message
import streamlit as st


class messageForRendering(TypedDict):
    role: str
    type: str
    content: str


def render_environment_profile(profile: EnvironmentProfile) -> None:
    # Render the codename as a subheader
    # Render the scenario with domain and realism in styled tags
    if len(profile.agent_goals) == 2:
        processed_agent1_goal = render_text_for_environment(
            profile.agent_goals[0]
        ).replace("\n", "<br>")
        processed_agent2_goal = render_text_for_environment(
            profile.agent_goals[1]
        ).replace("\n", "<br>")
    else:
        processed_agent1_goal = ""
        processed_agent2_goal = ""

    st.markdown(
        f"""
        <div style="background-color: #f9f9f9; padding: 10px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
            <p><strong>Scenario</strong>: {profile.scenario}</p>
            <div style="margin-top: 20px;">
                <div style="display: inline-block; width: 48%; vertical-align: top;">
                    <p><strong>Agent 1's Goal</strong></p>
                    <div style="background-color: #D1E9F6; padding: 10px; border-radius: 10px; margin-bottom: 5px;">
                        <p class="truncate">{processed_agent1_goal}</p>
                    </div>
                </div>
                <div style="display: inline-block; width: 48%; vertical-align: top;">
                    <p><strong>Agent 2's Goal</strong></p>
                    <div style="background-color: #D1E9F6; padding: 10px; border-radius: 10px; margin-bottom: 5px;">
                        <p class="truncate">{processed_agent2_goal}</p>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Foldable green container for additional information
    with st.expander("Additional Information", expanded=False):
        st.write(
            f"""
            <div style="background-color: #d0f5d0; padding: 10px; border-radius: 10px;">
                <h4>Primary Key</h4>
                <p>{profile.pk}</p>
                <h4>Codename</h4>
                <p>{profile.codename}</p>
                <h4>Source</h4>
                <p>{profile.source}</p>
                <h4>Relationship</h4>
                <p>{profile.relationship.name}</p>
                <h4>Age Constraint</h4>
                <p>{profile.age_constraint if profile.age_constraint else 'None'}</p>
                <h4>Occupation Constraint</h4>
                <p>{profile.occupation_constraint if profile.occupation_constraint else 'None'}</p>
                <h4>Agent Constraint</h4>
                <p>{profile.agent_constraint if profile.agent_constraint else 'None'}</p>
                <h4>Tag</h4>
                <p>{profile.tag}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def parse_reasoning(reasoning: str, num_agents: int) -> tuple[list[str], str]:
    """Parse the reasoning string into a dictionary."""
    sep_token = "SEPSEP"
    for i in range(1, num_agents + 1):
        reasoning = (
            reasoning.replace(f"Agent {i} comments:\n", sep_token)
            .strip(" ")
            .strip("\n")
        )
    all_chunks = reasoning.split(sep_token)
    general_comment = all_chunks[0].strip(" ").strip("\n")
    comment_chunks = all_chunks[-num_agents:]

    return comment_chunks, general_comment


def _map_gender_to_adj(gender: str) -> str:
    gender_to_adj = {
        "Man": "male",
        "Woman": "female",
        "Nonbinary": "nonbinary",
    }
    if gender:
        return gender_to_adj[gender]
    else:
        return ""


def compose_env_messages(env: ParallelSotopiaEnv) -> tuple[str, list[str]]:
    env_profile = env.profile
    env_to_render = env_profile.scenario
    goals_to_render = env_profile.agent_goals

    return env_to_render, goals_to_render


def render_for_humans(episode: EpisodeLog) -> list[messageForRendering]:
    """Generate a list of messages for human-readable version of the episode log."""

    messages_for_rendering: list[messageForRendering] = []

    for idx, turn in enumerate(episode.messages):
        is_observation_printed = False

        if idx == 0:
            assert (
                len(turn) >= 2
            ), "The first turn should have at least environment messages"

            messages_for_rendering.append(
                {"role": "Background Info", "type": "info", "content": turn[0][2]}
            )
            messages_for_rendering.append(
                {"role": "Background Info", "type": "info", "content": turn[1][2]}
            )
            messages_for_rendering.append(
                {"role": "System", "type": "divider", "content": "Start Simulation"}
            )

        for sender, receiver, message in turn:
            if not is_observation_printed and "Observation:" in message and idx != 0:
                extract_observation = message.split("Observation:")[1].strip()
                if extract_observation:
                    messages_for_rendering.append(
                        {
                            "role": "Observation",
                            "type": "observation",
                            "content": extract_observation,
                        }
                    )
                is_observation_printed = True

            if receiver == "Environment":
                if sender != "Environment":
                    if "did nothing" in message:
                        continue
                    elif "left the conversation" in message:
                        messages_for_rendering.append(
                            {
                                "role": "Environment",
                                "type": "leave",
                                "content": f"{sender} left the conversation",
                            }
                        )
                    else:
                        if "said:" in message:
                            message = message.split("said:")[1].strip()
                            messages_for_rendering.append(
                                {"role": sender, "type": "said", "content": message}
                            )
                        else:
                            message = message.replace("[action]", "")
                            messages_for_rendering.append(
                                {"role": sender, "type": "action", "content": message}
                            )
                else:
                    messages_for_rendering.append(
                        {
                            "role": "Environment",
                            "type": "environment",
                            "content": message,
                        }
                    )

    messages_for_rendering.append(
        {"role": "System", "type": "divider", "content": "End Simulation"}
    )

    reasoning_per_agent, general_comment = parse_reasoning(
        episode.reasoning,
        len(
            set(
                msg["role"]
                for msg in messages_for_rendering
                if msg["type"] in {"said", "action"}
            )
        ),
    )

    if general_comment == "":
        return messages_for_rendering[:-1]

    messages_for_rendering.append(
        {"role": "General", "type": "comment", "content": general_comment}
    )

    for idx, reasoning in enumerate(reasoning_per_agent):
        reasoning_lines = reasoning.split("\n")
        new_reasoning = ""
        for reasoning_line in reasoning_lines:
            dimension = reasoning_line.split(":")[0]
            new_reasoning += (
                (f"**{dimension}**: {':'.join(reasoning_line.split(':')[1:])}" + "\n")
                if dimension != ""
                else reasoning_line + "\n"
            )
        messages_for_rendering.append(
            {
                "role": f"Agent {idx + 1}",
                "type": "comment",
                "content": f"**Agent {idx + 1} reasoning**:\n{new_reasoning}\n\n**Rewards**: {str(episode.rewards[idx])}",
            }
        )

    for item in messages_for_rendering:
        item["content"] = item["content"].replace("$", "\\$")

    return messages_for_rendering


def compose_agent_messages(
    agents: Agents, target_agent_viewer: list[int] | None = None
) -> list[str]:
    if target_agent_viewer is None:
        target_agent_viewer = [idx + 1 for idx in range(len(agents))]

    agent_to_render = [
        render_text_for_agent(
            raw_text=_agent_profile_to_friendabove_self(
                agent.profile, agent_id + 1, display_name=False
            ),
            agent_id=target_agent_id,
        )
        for agent_id, (target_agent_id, agent) in enumerate(
            zip(target_agent_viewer, agents.values())
        )
    ]
    return agent_to_render


def render_messages(
    env: ParallelSotopiaEnv,
    agent_list: list[LLMAgent],
    messages: list[list[tuple[str, str, Message]]],
    reasoning: list[str],
    rewards: list[list[float]],
) -> list[messageForRendering]:
    epilog = EpisodeLog(
        environment=env.profile.pk,
        agents=[agent.profile.pk for agent in agent_list],
        tag="tmp",
        models=[env.model_name, agent_list[0].model_name, agent_list[1].model_name],
        messages=[
            [(m[0], m[1], m[2].to_natural_language()) for m in messages_in_turn]
            for messages_in_turn in messages
        ],
        reasoning=reasoning,
        rewards=rewards,
        rewards_prompt="",
    )
    rendered_messages = render_for_humans(epilog)
    return rendered_messages


def agent_profile_to_public_info(
    profile: AgentProfile, display_name: bool = False
) -> str:
    base_str: str = f"a {profile.age}-year-old {_map_gender_to_adj(profile.gender)} {profile.occupation.lower()}. {profile.gender_pronoun} pronouns. {profile.public_info} Personality and values description: {profile.personality_and_values}."
    if display_name:
        return f"{profile.first_name} {profile.last_name} is {base_str}"
    else:
        return base_str.capitalize()


def agent_profile_to_secret_info(
    profile: AgentProfile, display_name: bool = False
) -> str:
    if display_name:
        return f"{profile.first_name}'s secrets: {profile.secret}."
    else:
        return f"Secrets: {profile.secret}."


def get_public_info(profile: AgentProfile, display_name: bool = True) -> str:
    if profile.age == 0:
        return profile.public_info
    else:
        return agent_profile_to_public_info(profile, display_name=display_name)


def get_secret_info(profile: AgentProfile, display_name: bool = True) -> str:
    if profile.age == 0:
        return profile.secret
    else:
        return agent_profile_to_secret_info(profile, display_name=display_name)


def _agent_profile_to_friendabove_self(
    profile: AgentProfile, agent_id: int, display_name: bool = True
) -> str:
    return f"{get_public_info(profile=profile, display_name=display_name)}<p viewer='agent_{agent_id}'>{get_secret_info(profile=profile, display_name=display_name)}</p>"
