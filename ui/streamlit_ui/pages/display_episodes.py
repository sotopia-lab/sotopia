import streamlit as st
from ui.streamlit_ui.rendering import (
    render_environment_profile,
    render_conversation_and_evaluation,
)
from sotopia.database import EpisodeLog, EnvironmentProfile
import requests

st.title("Episode")

st.write("Here are some instructions about using the episode renderer.")


def render_episodes() -> None:
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
            episode = st.session_state.current_episodes[selected_index]
            response = requests.get(
                f"{st.session_state.API_BASE}/scenarios/id/{episode.environment}"
            )  # return a list of scenarios
            scenario = response.json() if response.status_code == 200 else []
            scenario = scenario[0]
            environment_profile = EnvironmentProfile(**scenario)
            render_environment_profile(environment_profile)

    st.markdown("---")
    render_conversation_and_evaluation(episode)


render_episodes()
