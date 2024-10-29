import os

import streamlit as st

from sotopia.ui.socialstream.chat import chat_demo_omniscient, chat_demo_simple
from sotopia.ui.socialstream.rendering import rendering_demo
from sotopia.ui.socialstream.utils import initialize_session_state, reset_database


def update_database_callback() -> None:
    new_database_url = st.session_state.new_database_url
    updated_url = (
        new_database_url if new_database_url != "" else st.session_state.DEFAULT_DB_URL
    )
    try:
        reset_database(updated_url)
    except Exception as e:
        st.error(f"Error occurred while updating database: {e}, please try again.")

    st.session_state.current_database_url = updated_url
    initialize_session_state(force_reload=True)

    print("Updated DB URL: ", st.session_state.current_database_url)


st.set_page_config(page_title="SocialStream_Demo", page_icon="🧊", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        max-width: 2500px;
    }
    """,
    unsafe_allow_html=True,
)  # set the sidebar width to be wider

DISPLAY_MODE = "Display Episodes"
CHAT_SIMPLE_MODE = "Simple Chat"
CHAT_OMNISCIENT_MODE = "Omniscient Chat & Editable Scenario"

if "DEFAULT_DB_URL" not in st.session_state:
    st.session_state.DEFAULT_DB_URL = os.environ.get("REDIS_OM_URL", "")
    st.session_state.current_database_url = st.session_state.DEFAULT_DB_URL
    print("Default DB URL: ", st.session_state.DEFAULT_DB_URL)

# impl 1: use sidebar to update URL
new_database_url = st.sidebar.text_input(
    "Enter Database URL: (Optional, starting in redis://)",
    value="",
    on_change=update_database_callback,
    key="new_database_url",
)

# # impl 2: use query params in URL
# query_params = st.experimental_get_query_params()
# current_database_url = query_params.get('database', [''])[0]

# def get_actual_database_url() -> str:
#     return current_database_url or st.session_state.DEFAULT_DB_URL

# actual_database_url = get_actual_database_url()
# if st.session_state.current_database_url != actual_database_url:
#     st.session_state.current_database_url = actual_database_url
#     reset_database(actual_database_url)
#     initialize_session_state(force_reload=True)
#     print("Actual DB URL: ", actual_database_url)
#     st.rerun()

option = st.sidebar.radio(
    "Function", (DISPLAY_MODE, CHAT_SIMPLE_MODE, CHAT_OMNISCIENT_MODE)
)
if option != st.session_state.get("mode", None):
    # when switching between modes, reset the active agent
    if "active" in st.session_state:
        del st.session_state["active"]

if option == DISPLAY_MODE:
    st.session_state.mode = DISPLAY_MODE
    rendering_demo()
elif option == CHAT_SIMPLE_MODE:
    st.session_state.editable = False
    st.session_state.mode = CHAT_SIMPLE_MODE
    chat_demo_simple()
elif option == CHAT_OMNISCIENT_MODE:
    st.session_state.mode = CHAT_OMNISCIENT_MODE
    chat_demo_omniscient()
