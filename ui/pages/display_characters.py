import streamlit as st
import random
from sotopia.database import BaseAgentProfile

# Importing avatars
from ui.rendering import render_character, local_css, get_agents


local_css("./css/style.css")


def display_characters() -> None:
    st.title("Characters")
    all_characters = get_agents()
    # Randomize the order of characters
    all_characters = dict(sorted(all_characters.items(), key=lambda _: random.random()))
    col1, col2 = st.columns(2, gap="medium")
    for i, (name, character) in enumerate(all_characters.items()):
        with col1 if i % 2 == 0 else col2:
            character_profile = BaseAgentProfile(**character)
            render_character(character_profile)
            st.write("---")


display_characters()
