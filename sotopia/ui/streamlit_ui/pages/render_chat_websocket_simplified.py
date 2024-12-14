import asyncio
import json
from typing import Any

import aiohttp
import requests
import streamlit as st
import time

from sotopia.ui.streamlit_ui.utils import get_abstract
from sotopia.ui.streamlit_ui.rendering.render_episode import rendering_episode_full
from sotopia.database import EpisodeLog


def compose_agent_names(agent_dict: dict[Any, Any]) -> str:
    return f"{agent_dict['first_name']} {agent_dict['last_name']}"


def get_scenarios() -> dict[str, dict[Any, Any]]:
    with requests.get(f"{st.session_state.API_BASE}/scenarios") as resp:
        scenarios = resp.json()
    return {scenario["codename"]: scenario for scenario in scenarios}


def get_agents() -> tuple[dict[str, dict[Any, Any]], dict[str, dict[Any, Any]]]:
    with requests.get(f"{st.session_state.API_BASE}/agents") as resp:
        agents = resp.json()
    return {compose_agent_names(agent): agent for agent in agents}, {
        compose_agent_names(agent): agent for agent in agents
    }


def get_models() -> tuple[dict[str, dict[Any, Any]], dict[str, dict[Any, Any]]]:
    with requests.get(f"{st.session_state.API_BASE}/models") as resp:
        models = resp.json()
    return {model: model for model in models}, {model: model for model in models}


def initialize_session_state() -> None:
    if "active" not in st.session_state:
        # Initialize base state
        st.session_state.scenarios = get_scenarios()
        st.session_state.agent_list_1, st.session_state.agent_list_2 = get_agents()
        st.session_state.agent_model_1, st.session_state.agent_model_2 = get_models()

        # Use first items as default choices
        st.session_state.scenario_choice = list(st.session_state.scenarios.keys())[0]
        st.session_state.agent_choice_1 = list(st.session_state.agent_list_1.keys())[0]
        st.session_state.agent_choice_2 = list(st.session_state.agent_list_2.keys())[0]
        st.session_state.agent1_model_choice = list(
            st.session_state.agent_model_1.keys()
        )[0]
        st.session_state.agent2_model_choice = list(
            st.session_state.agent_model_2.keys()
        )[0]

        # Initialize websocket manager and message list
        st.session_state.messages = []
        # Set initial active state
        st.session_state.active = False

        print("Session state initialized")


chat_history_container = st.empty()


def parse_messages(messages: list[str]) -> list[dict[str, Any]]:
    chat_messages = []
    messages = [message for message in messages[1:] if message != ""]
    evaluation_available = len(messages) >= 2 and "Agent 1 comments" in messages[-2]
    evaluations = [] if not evaluation_available else messages[-2:]
    conversations = messages if not evaluation_available else messages[:-2]

    chat_messages = [
        {"role": "human", "content": message} for message in conversations
    ] + [
        {
            "role": "assistant",
            "content": message,
        }
        for message in evaluations
    ]

    return chat_messages


async def run_simulation() -> None:
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            f"{st.session_state.WS_BASE}/ws/simulation?token=demo-token"
        ) as ws:
            await ws.send_str(
                json.dumps(
                    {
                        "type": "START_SIM",
                        "data": {
                            "env_id": st.session_state.scenarios[
                                st.session_state.scenario_choice
                            ]["pk"],
                            "agent_ids": [
                                st.session_state.agent_list_1[
                                    st.session_state.agent_choice_1
                                ]["pk"],
                                st.session_state.agent_list_2[
                                    st.session_state.agent_choice_2
                                ]["pk"],
                            ],
                            "agent_models": [
                                st.session_state.agent_model_1[
                                    st.session_state.agent_model_choice_1
                                ],
                                st.session_state.agent_model_2[
                                    st.session_state.agent_model_choice_2
                                ],
                            ],
                        },
                    }
                )
            )

            while True:
                msg = await ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    message = json.loads(msg.data)
                    if message["type"] == "END_SIM":
                        break

                    if message["data"]["type"] == "messages":
                        epilog = EpisodeLog(**message["data"]["messages"])
                        st.session_state.messages = epilog
                        with chat_history_container.container():
                            rendering_episode_full(epilog)

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

                if st.session_state.get("stop_sim", False):
                    await ws.send_str(json.dumps({"type": "FINISH_SIM", "data": ""}))


def start_callback() -> None:
    if st.session_state.agent_choice_1 == st.session_state.agent_choice_2:
        st.error("Please select different agents")
    else:
        st.session_state.active = True
        st.session_state.stop_sim = False
        st.session_state.messages = None
        try:
            asyncio.run(run_simulation())
        except Exception as e:
            st.error(f"Error running simulation: {e}")
        finally:
            st.session_state.active = False


def stop_callback() -> None:
    st.session_state.stop_sim = True


def is_active() -> bool:
    assert isinstance(st.session_state.active, bool)
    return st.session_state.active


def chat_demo() -> None:
    initialize_session_state()

    with st.sidebar:
        with st.container():
            # Scenario and Agent Selection
            with st.expander("Simulation Setup", expanded=True):
                scenario_col, scenario_desc_col = st.columns(2)
                with scenario_col:
                    st.selectbox(
                        "Choose a scenario:",
                        st.session_state.scenarios.keys(),
                        key="scenario_choice",
                        disabled=is_active(),
                    )

                with scenario_desc_col:
                    st.markdown(
                        f"""**Description:** {get_abstract(st.session_state.scenarios[st.session_state.scenario_choice]["scenario"])}""",
                        unsafe_allow_html=True,
                    )

                agent1_col, agent2_col = st.columns(2)
                with agent1_col:
                    st.selectbox(
                        "Choose Agent 1:",
                        st.session_state.agent_list_1.keys(),
                        key="agent_choice_1",
                        disabled=is_active(),
                    )

                with agent2_col:
                    st.selectbox(
                        "Choose Agent 2:",
                        st.session_state.agent_list_2.keys(),
                        key="agent_choice_2",
                        disabled=is_active(),
                    )

                model1_col, model2_col = st.columns(2)
                with model1_col:
                    st.selectbox(
                        "Choose Agent 1 Model:",
                        st.session_state.agent_model_1.keys(),
                        key="agent_model_choice_1",
                        disabled=is_active(),
                    )

                with model2_col:
                    st.selectbox(
                        "Choose Agent 2 Model:",
                        st.session_state.agent_model_2.keys(),
                        key="agent_model_choice_2",
                        disabled=is_active(),
                    )

        # Control Buttons
        col1, col2 = st.columns([1, 1])

        with col1:
            st.button(
                "Start Simulation",
                disabled=is_active(),
                on_click=start_callback,
            )

        with col2:
            st.button(
                "Stop Simulation",
                disabled=not is_active(),
                on_click=stop_callback,
            )

        # if is_active():
        #     try:
        #         asyncio.run(run_simulation())
        #     except Exception as e:
        #         st.error(f"Error running simulation: {e}")
        #     finally:
        #         st.session_state.active = False

        while is_active() and st.session_state.messages:
            with chat_history_container.container():
                rendering_episode_full(st.session_state.messages)
            time.sleep(1)

        if st.session_state.messages:
            with chat_history_container.container():
                rendering_episode_full(st.session_state.messages)


chat_demo()
