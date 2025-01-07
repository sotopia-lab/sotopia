import streamlit as st
import requests

from sotopia.database import BaseAgentProfile
from ui.rendering import local_css

# add fields for agent profiles


def rendering_character_form() -> None:
    local_css("././css/style.css")
    st.markdown("<h1>Character Creation</h1>", unsafe_allow_html=True)
    st.write("Fill in the fields below to create a new character:")

    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    age = st.number_input("Age", min_value=0)
    occupation = st.text_input("Occupation")
    gender = st.text_input("Gender")
    gender_pronoun = st.text_input("Gender Pronoun")
    public_info = st.text_area("Public Info")

    if st.button("Create Character"):
        agent_profile = BaseAgentProfile(
            first_name=first_name,
            last_name=last_name,
            age=age,
            occupation=occupation,
            gender=gender,
            gender_pronoun=gender_pronoun,
            public_info=public_info,
            tag="customized_agent",
        )
        print(agent_profile)

        response = requests.post(
            f"{st.session_state.API_BASE}/agents",
            json=agent_profile.model_dump(),
        )

        if response.status_code != 200:
            st.error("Failed to create character. Error message: " + response.text)

        else:
            agent_id = response.json()
            st.success("Character created successfully! ID: " + agent_id)
            retrieved_agent = requests.get(
                f"{st.session_state.API_BASE}/agents/id/{agent_id}"
            )
            st.write(retrieved_agent.json())


rendering_character_form()
