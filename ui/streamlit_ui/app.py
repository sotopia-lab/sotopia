import os

import streamlit as st

# Page Configuration
st.set_page_config(page_title="SocialStream_Demo", page_icon="🧊", layout="wide")

# PORT = 8800
# st.session_state.API_BASE = f"http://localhost:{PORT}"
# st.session_state.WS_BASE = f"ws://localhost:{PORT}"

DEFAULT_BASE = "sotopia-lab--sotopia-fastapi-webapi-serve.modal.run"

# Modal Configuration

if "API_BASE" not in st.session_state:
    st.session_state.API_BASE = f"https://{DEFAULT_BASE}"
    st.session_state.WS_BASE = f"ws://{DEFAULT_BASE}"


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


base_path = os.path.dirname(os.path.abspath(__file__))
page_path = os.path.join(base_path, "pages")

display_intro = st.Page(
    f"{page_path}/intro.py", title="Introduction", icon=":material/home:"
)

display_scenarios = st.Page(
    f"{page_path}/display_scenarios.py",
    title="Scenarios",
    icon=":material/insert_drive_file:",
)
display_characters = st.Page(
    f"{page_path}/display_characters.py", title="Characters", icon=":material/people:"
)
display_episodes = st.Page(
    f"{page_path}/display_episodes.py", title="Episode", icon=":material/photo_library:"
)

display_chat = st.Page(
    f"{page_path}/render_chat_websocket.py",
    title="Simulation",
    # icon=":material/add:",
)

display_evaluation_dimensions = st.Page(
    f"{page_path}/display_evaluation_dimensions.py",
    title="Evaluation Dimensions",
    # icon=":material/add:",
)

add_characters = st.Page(
    f"{page_path}/add_characters.py", title="Add Characters", icon=":material/add:"
)

add_scenarios = st.Page(
    f"{page_path}/add_scenarios.py", title="Add Scenarios", icon=":material/add:"
)
add_evaluation_dimensions = st.Page(
    f"{page_path}/add_evaluation_dimension.py",
    title="Add Evaluation Dimensions",
    icon=":material/add:",
)

pg = st.navigation(
    [
        display_intro,
        display_scenarios,
        display_characters,
        display_episodes,
        display_chat,
        display_evaluation_dimensions,
        add_characters,
        add_scenarios,
        add_evaluation_dimensions,
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

with st.sidebar:
    with st.expander("API Configuration", expanded=False):
        st.session_state.API_BASE = st.text_input(
            "API Base URL:",
            value=st.session_state.API_BASE,
            placeholder="Enter API base URL",
        )

        st.session_state.WS_BASE = st.text_input(
            "WebSocket Base URL:",
            value=st.session_state.WS_BASE,
            placeholder="Enter WebSocket base URL",
        )

        # Optional: Add a reset button
        if st.button("Reset to Default"):
            st.session_state.API_BASE = f"https://{DEFAULT_BASE}"
            st.session_state.WS_BASE = f"ws://{DEFAULT_BASE}"
            st.rerun()


pg.run()
