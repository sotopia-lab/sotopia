import streamlit as st

from ..rendering import rendering_episode

from ..utils import initialize_session_state


def local_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("./css/style.css")


def display_scenarios() -> None:
    initialize_session_state()
    st.title("Scenarios")
    all_episodes = st.session_state.current_episodes

    col1, col2 = st.columns(2, gap="medium")
    for i, episode in enumerate(all_episodes):
        with col1 if i % 2 == 0 else col2:
            rendering_episode(episode)
            st.write("---")


display_scenarios()
