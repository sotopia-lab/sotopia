import streamlit as st

from sotopia.database import AgentProfile

# Importing avatars
from sotopia.ui.streamlit_ui.rendering import render_character, local_css, get_agents


local_css("./css/style.css")


def display_characters() -> None:
    st.title("Characters")
    all_characters = get_agents()
    col1, col2 = st.columns(2, gap="medium")
    for i, (name, character) in enumerate(all_characters.items()):
        with col1 if i % 2 == 0 else col2:
            character_profile = AgentProfile(**character)
            render_character(character_profile)
            st.write("---")


display_characters()
