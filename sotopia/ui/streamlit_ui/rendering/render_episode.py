import json

import streamlit as st
from sotopia.database import AgentProfile, EnvironmentProfile, EpisodeLog
from sotopia.envs.parallel import render_text_for_agent, render_text_for_environment
from sotopia.ui.streamlit_ui.pages.display_characters import get_avatar_icons

from sotopia.ui.streamlit_ui.rendering.rendering_utils import (
    _agent_profile_to_friendabove_self,
    render_for_humans,
)
from sotopia.ui.streamlit_ui.utils import (
    get_full_name,
)


role_mapping = {
    "Background Info": "background",
    "System": "info",
    "Environment": "env",
    "Observation": "obs",
    "General": "eval",
}


def local_css(file_name: str) -> None:
    with open(file_name) as f:
        # print("\n\n STYLING", f.read())
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def update_database_callback() -> None:
    pass


def rendering_episode(episode: EpisodeLog) -> None:
    local_css("./././css/style.css")

    agents = [AgentProfile.get(agent) for agent in episode.agents]
    agent_names = [get_full_name(agent) for agent in agents]
    environment = EnvironmentProfile.get(episode.environment)
    agent_goals = [
        render_text_for_agent(agent_goal, agent_id)
        for agent_id, agent_goal in enumerate(environment.agent_goals)
    ]

    st.markdown(
        f"""
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
        <p><strong>Scenario</strong>: { render_text_for_environment(environment.scenario)}</p>
        <div style="margin-top: 20px;">
            <div style="display: inline-block; width: 48%; vertical-align: top;">
                <p><strong>Agent 1's Goal</strong></p>
                <div style="background-color: #D1E9F6; padding: 10px; border-radius: 10px; margin-bottom: 5px;">
                    <p class="truncate">{agent_goals[0]}</p>
                </div>
            </div>
            <div style="display: inline-block; width: 48%; vertical-align: top;">
                <p><strong>Agent 2's Goal</strong></p>
                <div style="background-color: #D1E9F6; padding: 10px; border-radius: 10px; margin-bottom: 5px;">
                    <p class="truncate">{agent_goals[1]}</p>
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    with st.expander("Additional Information", expanded=False):
        info_col1, info_col2 = st.columns(2)
        info_1 = _agent_profile_to_friendabove_self(agents[0], agent_id=1)
        info_2 = _agent_profile_to_friendabove_self(agents[1], agent_id=2)
        with info_col1:
            st.write(
                f"""
            <div style="background-color: #d0f5d0; padding: 10px; border-radius: 10px;">
                <p> <strong>{agent_names[0]}'s info </strong></h4>
                <p>{info_1}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with info_col2:
            st.write(
                f"""
            <div style="background-color: #d0f5d0; padding: 10px; border-radius: 10px; margin-bottom: 12px;">
                <p> <strong>{agent_names[1]}'s info </strong></h4>
                <p>{info_2}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )


def rendering_episode_full(episode: EpisodeLog) -> None:
    print("Episode: ", episode)
    agents = [AgentProfile.get(agent) for agent in episode.agents]
    agent_names = [get_full_name(agent) for agent in agents]
    environment = EnvironmentProfile.get(episode.environment)

    avatar_mapping = {
        agent_names[0]: "🧔🏻",
        agent_names[1]: "🧑",
    }

    messages = render_for_humans(episode)

    background_messages = [
        message for message in messages if message["role"] == "Background Info"
    ]
    evaluation_messages = [
        message for message in messages if message["type"] == "comment"
    ]
    conversation_messages = [
        message
        for message in messages
        if message not in background_messages and message not in evaluation_messages
    ]

    assert (
        len(background_messages) == 2
    ), f"Need 2 background messages, but got {len(background_messages)}"

    print(f"\n\ENVIRONMENT {environment}")

    rendering_episode(episode)

    st.markdown("---")

    st.subheader("Conversation & Evaluation")
    with st.expander("Conversation", expanded=True):
        for index, message in enumerate(conversation_messages):
            role = role_mapping.get(message["role"], message["role"])
            content = message["content"]

            if role == "obs" or message.get("type") == "action":
                try:
                    content = json.loads(content)
                except Exception as e:
                    print(e)

            with st.chat_message(
                role, avatar=avatar_mapping.get(message["role"], None)
            ):
                if isinstance(content, dict):
                    st.json(content)
                elif role == "info":
                    st.markdown(
                        f"""
                        <div style="background-color: lightblue; padding: 10px; border-radius: 5px;">
                            {content}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    if role == agent_names[1]:
                        # add background grey color
                        st.write(f"**{role}**")
                        st.markdown(
                            f"""
                            <div style="background-color: lightgrey; padding: 10px; border-radius: 5px;">
                                {content}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.write(f"**{role}**")
                        st.markdown(
                            content.replace("\n", "<br />"), unsafe_allow_html=True
                        )

    with st.expander("Evaluation Results", expanded=True):
        for message in evaluation_messages:
            st.markdown(
                message["content"].replace("\n", "<br />"), unsafe_allow_html=True
            )


def rendering_episodes() -> None:
    local_css("./././css/style.css")

    tags = [
        "gpt-4_gpt-4_v0.0.1_clean",
        "1019_hiring_equal_cooperative_salary_start_date_trust-bigfive-low_transparency-high_competence-low_adaptability-Agreeableness",
    ]
    if "current_episodes" not in st.session_state:
        st.session_state["current_episodes"] = EpisodeLog.find(
            EpisodeLog.tag == tags[0]
        ).all()

    def update() -> None:
        tag = st.session_state.selected_tag
        st.session_state.current_episodes = EpisodeLog.find(EpisodeLog.tag == tag).all()

    with st.container():
        # Dropdown for codename selection
        st.selectbox(
            "Choose a tag:",
            tags,
            index=0,
            on_change=update,
            key="selected_tag",
        )

        selected_index = st.number_input(
            "Specify the index of the episode to display:",
            min_value=0,
            max_value=len(st.session_state.current_episodes) - 1,
            value=0,
            step=1,
        )

        if selected_index < len(st.session_state.current_episodes):
            # TODO unify the display function across render and chat
            episode = st.session_state.current_episodes[selected_index]
            agents = [AgentProfile.get(agent) for agent in episode.agents]
            agent_names = [get_full_name(agent) for agent in agents]
            environment = EnvironmentProfile.get(episode.environment)

            # avatar_mapping = {
            #     agent_names[0]: "🧔🏻",
            #     agent_names[1]: "🧑",
            # }

            avatar_mapping = get_avatar_icons()

            messages = render_for_humans(episode)

            background_messages = [
                message for message in messages if message["role"] == "Background Info"
            ]
            evaluation_messages = [
                message for message in messages if message["type"] == "comment"
            ]
            conversation_messages = [
                message
                for message in messages
                if message not in background_messages
                and message not in evaluation_messages
            ]

            assert (
                len(background_messages) == 2
            ), f"Need 2 background messages, but got {len(background_messages)}"

            print(f"\n\ENVIRONMENT {environment}")

            rendering_episode(episode)

    st.markdown("---")

    st.subheader("Conversation & Evaluation")
    with st.expander("Conversation", expanded=True):
        for index, message in enumerate(conversation_messages):
            role = role_mapping.get(message["role"], message["role"])
            content = message["content"]

            if role == "obs" or message.get("type") == "action":
                try:
                    content = json.loads(content)
                except Exception as e:
                    print(e)

            avatar_path = avatar_mapping.get(message["role"], None)
            with st.chat_message(role, avatar=avatar_path):
                if isinstance(content, dict):
                    st.json(content)
                elif role == "info":
                    st.markdown(
                        f"""
                        <div style="background-color: lightblue; padding: 10px; border-radius: 5px;">
                            {content}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    if role == agent_names[1]:
                        # add background grey color
                        st.write(f"**{role}**")
                        st.markdown(
                            f"""
                            <div style="background-color: lightgrey; padding: 10px; border-radius: 5px;">
                                {content}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.write(f"**{role}**")
                        st.markdown(
                            content.replace("\n", "<br />"), unsafe_allow_html=True
                        )

    with st.expander("Evaluation Results", expanded=True):
        for message in evaluation_messages:
            st.markdown(
                message["content"].replace("\n", "<br />"), unsafe_allow_html=True
            )
