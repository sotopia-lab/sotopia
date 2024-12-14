import streamlit as st
from sotopia.ui.streamlit_ui.rendering import (
    render_environment_profile,
    local_css,
    get_scenarios,
)
from sotopia.database import EnvironmentProfile


local_css("./css/style.css")


def display_scenarios() -> None:
    st.title("Scenarios")
    scenarios = get_scenarios()
    col1, col2 = st.columns(2, gap="medium")
    for index, (codename, scenario) in enumerate(scenarios.items()):
        with col1 if index % 2 == 0 else col2:
            environment_profile = EnvironmentProfile(**scenario)
            render_environment_profile(environment_profile)
            st.write("---")


display_scenarios()
