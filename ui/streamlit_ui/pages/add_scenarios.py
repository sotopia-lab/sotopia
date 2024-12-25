"""
Definition
@app.post("/scenarios/", response_model=str)
async def create_scenario(scenario: EnvironmentProfileWrapper) -> str:
    scenario_profile = EnvironmentProfile(**scenario.model_dump())
    scenario_profile.save()
    pk = scenario_profile.pk
    assert pk is not None
    return pk

class EnvironmentProfileWrapper(BaseModel):
    Wrapper for EnvironmentProfile to avoid pydantic v2 issues

    codename: str
    source: str = ""
    scenario: str = ""
    agent_goals: list[str] = []
    relationship: Literal[0, 1, 2, 3, 4, 5] = 0
    age_constraint: str | None = None
    occupation_constraint: str | None = None
    agent_constraint: list[list[str]] | None = None
    tag: str = ""

class RelationshipType(IntEnum):
    stranger = 0
    know_by_name = 1
    acquaintance = 2
    friend = 3
    romantic_relationship = 4
    family_member = 5

The age constraint of the environment, a list of tuples, each tuple is a range of age, e.g., '[(18, 25), (30, 40)]'
means the environment is only available to agent one between 18 and 25, and agent two between 30 and 40
"""

import streamlit as st
from ui.streamlit_ui.rendering import local_css
from sotopia.ui.fastapi_server import EnvironmentProfileWrapper
from sotopia.database import RelationshipType
import requests


def rendering_scenario_form() -> None:
    local_css("././css/style.css")
    st.markdown("<h1>Scenario Creation</h1>", unsafe_allow_html=True)

    codename = st.text_input("Codename")
    source = st.text_input("Source")
    scenario = st.text_area("Scenario")
    # present relationship type with the descriptions, only accept one choice, then map to the enum
    relationship_mapping = {
        "Stranger": RelationshipType.stranger,
        "Know by Name": RelationshipType.know_by_name,
        "Acquaintance": RelationshipType.acquaintance,
        "Friend": RelationshipType.friend,
        "Romantic Relationship": RelationshipType.romantic_relationship,
        "Family Member": RelationshipType.family_member,
    }

    selected_relationship = st.selectbox(
        "Relationship",
        list(relationship_mapping.keys()),
    )
    relationship = relationship_mapping[selected_relationship]

    # first choose whether to use age constraint and then choose age range
    use_age_constraint = st.checkbox("Use Age Constraint")
    if use_age_constraint:
        min_age = st.number_input("Min Age", min_value=0, max_value=100)
        max_age = st.number_input("Max Age", min_value=0, max_value=100)
        age_constraint = f"[({min_age}, {max_age})]"
        if min_age > max_age:
            st.error("Min age cannot be greater than max age")
    else:
        age_constraint = None

    # first choose whether to use occupation constraint and then choose occupation
    use_occupation_constraint = st.checkbox(
        "Use Occupation Constraint, use comma to separate multiple occupations"
    )
    if use_occupation_constraint:
        occupation = st.text_input("Occupation")
        occupation_constraint = f"[{occupation}]"
    else:
        occupation_constraint = None

    agent1_goal = st.text_input("Agent 1 Goal")
    agent2_goal = st.text_input("Agent 2 Goal")

    if st.button("Create Scenario"):
        scenario_profile = EnvironmentProfileWrapper(
            codename="[Customize]" + codename,
            source=source,
            scenario=scenario,
            relationship=relationship,
            age_constraint=age_constraint,
            occupation_constraint=occupation_constraint,
            agent_goals=[agent1_goal, agent2_goal],
            tag="customized_scenario",
        )

        response = requests.post(
            f"{st.session_state.API_BASE}/scenarios/",
            json=scenario_profile.model_dump(),
        )

        if response.status_code != 200:
            st.error("Failed to create scenario. Error: " + response.text)
        else:
            # there are quotes in the response
            scenario_id = response.json()

            st.success("Scenario created successfully! Scenario ID: " + scenario_id)
            retrieved_scenario = requests.get(
                f"{st.session_state.API_BASE}/scenarios/id/{scenario_id}"
            )
            st.write(retrieved_scenario.json())


rendering_scenario_form()
