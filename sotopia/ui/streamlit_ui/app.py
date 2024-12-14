import os

import streamlit as st

# Page Configuration
st.set_page_config(page_title="SocialStream_Demo", page_icon="🧊", layout="wide")

# PORT = 8800
# st.session_state.API_BASE = f"http://localhost:{PORT}"
# st.session_state.WS_BASE = f"ws://localhost:{PORT}"

# Modal Configuration
st.session_state.API_BASE = (
    "https://sotopia-lab--sotopia-fastapi-webapi-serve.modal.run"
)
st.session_state.WS_BASE = "ws://sotopia-lab--sotopia-fastapi-webapi-serve.modal.run"


def update_database_callback() -> None:
    new_database_url = st.session_state.new_database_url
    updated_url = (
        new_database_url if new_database_url != "" else st.session_state.DEFAULT_DB_URL
    )
    try:
        pass
    except Exception as e:
        st.error(f"Error occurred while updating database: {e}, please try again.")

    st.session_state.current_database_url = updated_url
    print("Updated DB URL: ", st.session_state.current_database_url)


display_intro = st.Page(
    "./pages/intro.py", title="Introduction", icon=":material/home:"
)
display_episodes = st.Page(
    "./pages/display_episodes.py", title="Episode", icon=":material/photo_library:"
)
display_scenarios = st.Page(
    "./pages/display_scenarios.py",
    title="Scenarios",
    icon=":material/insert_drive_file:",
)
display_characters = st.Page(
    "./pages/display_characters.py", title="Characters", icon=":material/people:"
)

display_chat = st.Page(
    "./pages/render_chat_websocket.py",
    title="Simulation",
    icon=":material/add:",
)

add_characters = st.Page(
    "./pages/add_characters.py", title="Add Characters", icon=":material/add:"
)

add_scenarios = st.Page(
    "./pages/add_scenarios.py", title="Add Scenarios", icon=":material/add:"
)

pg = st.navigation(
    [
        display_intro,
        display_scenarios,
        display_episodes,
        display_characters,
        display_chat,
        add_characters,
        add_scenarios,
    ]
)

# Reset active agent when switching modes across pages
if "mode" not in st.session_state or pg.title != st.session_state.get("mode", None):
    if "active" in st.session_state:
        del st.session_state["active"]
        # print("Active agent reset.")

    st.session_state.mode = pg.title


# DB URL Configuration
if "DEFAULT_DB_URL" not in st.session_state:
    st.session_state.DEFAULT_DB_URL = os.environ.get("REDIS_OM_URL", "")
    st.session_state.current_database_url = st.session_state.DEFAULT_DB_URL
    print("Default DB URL: ", st.session_state.DEFAULT_DB_URL)

# impl 2: popup update URL
with st.sidebar.popover("(Optional) Enter Database URL"):
    new_database_url = st.text_input(
        "URL: (starting in redis://)",
        value="",
        on_change=update_database_callback,
        key="new_database_url",
    )


pg.run()
