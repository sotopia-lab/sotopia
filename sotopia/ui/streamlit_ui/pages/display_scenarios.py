import requests
import streamlit as st
from sotopia.ui.streamlit_ui.rendering.rendering_utils import render_environment_profile
from sotopia.database import EnvironmentProfile


def local_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("./css/style.css")


def display_scenarios() -> None:
    st.title("Scenarios")

    response = requests.get(f"{st.session_state.API_BASE}/scenarios")
    scenarios = response.json() if response.status_code == 200 else []

    col1, col2 = st.columns(2, gap="medium")
    for i, scenario in enumerate(scenarios):
        with col1 if i % 2 == 0 else col2:
            environment_profile = EnvironmentProfile(**scenario)
            render_environment_profile(environment_profile)
            st.write("---")


display_scenarios()
