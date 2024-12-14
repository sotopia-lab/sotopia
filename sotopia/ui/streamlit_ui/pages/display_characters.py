import streamlit as st

from sotopia.database import AgentProfile

# Importing avatars
from sotopia.ui.streamlit_ui.rendering import render_character


def local_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def display_characters() -> None:
    st.title("Characters")
    all_characters = AgentProfile.find().all()

    col1, col2 = st.columns(2, gap="medium")
    for i, character in enumerate(all_characters):
        with col1 if i % 2 == 0 else col2:
            assert isinstance(character, AgentProfile)
            render_character(character)
            st.write("---")


display_characters()
