import asyncio
import json
from typing import Any
import time
import aiohttp
import requests
import streamlit as st


def get_scenarios() -> dict[str, dict[Any, Any]]:
    with requests.get(f"{st.session_state.API_BASE}/scenarios") as resp:
        scenarios = resp.json()
    return {scenario["codename"]: scenario for scenario in scenarios}


def get_agents() -> tuple[dict[str, dict[Any, Any]], dict[str, dict[Any, Any]]]:
    with requests.get(f"{st.session_state.API_BASE}/agents") as resp:
        agents = resp.json()
    return {f"{agent['first_name']} {agent['last_name']}": agent for agent in agents}, {
        f"{agent['first_name']} {agent['last_name']}": agent for agent in agents
    }


def get_models() -> tuple[dict[str, dict[Any, Any]], dict[str, dict[Any, Any]]]:
    with requests.get(f"{st.session_state.API_BASE}/models") as resp:
        models = resp.json()
    return {model: model for model in models}, {model: model for model in models}


async def websocket_client(url: str, initial_message: dict, timeout: int = 300) -> None:
    start_time = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                # Send initial message
                await ws.send_str(json.dumps(initial_message))

                # Receive messages until timeout
                while time.time() - start_time < timeout:
                    try:
                        msg = await ws.receive(timeout=1.0)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            handle_message(data)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"Error receiving message: {e}")
                        break
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        st.session_state.active = False


def initialize_session_state() -> None:
    if "active" not in st.session_state:
        st.session_state.scenarios = get_scenarios()
        st.session_state.agent_list_1, st.session_state.agent_list_2 = get_agents()
        st.session_state.agent_model_1, st.session_state.agent_model_2 = get_models()

        st.session_state.scenario_choice = list(st.session_state.scenarios.keys())[0]
        st.session_state.agent_choice_1 = list(st.session_state.agent_list_1.keys())[0]
        st.session_state.agent_choice_2 = list(st.session_state.agent_list_2.keys())[0]
        st.session_state.agent1_model_choice = list(
            st.session_state.agent_model_1.keys()
        )[0]
        st.session_state.agent2_model_choice = list(
            st.session_state.agent_model_2.keys()
        )[0]

        st.session_state.messages = []
        st.session_state.active = False


def handle_message(message: dict[str, Any]) -> None:
    match message["type"]:
        case "END_SIM":
            st.session_state.active = False
            st.rerun()
        case "SERVER_MSG":
            st.session_state.messages = message["data"]
            # st.session_state.messages.append({
            #     "role": message["data"]["role"],
            #     "content": message["data"]["content"],
            #     "type": message["data"]["type"]
            # })
        case "ERROR":
            print(f"Error in message: {message['data']}")
            st.error(f"Error: {message['data']}")


def start_simulation() -> None:
    if st.session_state.agent_choice_1 == st.session_state.agent_choice_2:
        st.error("Please select different agents")
        return

    st.session_state.active = True
    st.session_state.messages = []

    initial_message = {
        "type": "START_SIM",
        "data": {
            "env_id": st.session_state.scenarios[st.session_state.scenario_choice][
                "pk"
            ],
            "agent_ids": [
                st.session_state.agent_list_1[st.session_state.agent_choice_1]["pk"],
                st.session_state.agent_list_2[st.session_state.agent_choice_2]["pk"],
            ],
            "agent_models": [
                st.session_state.agent_model_1[st.session_state.agent1_model_choice],
                st.session_state.agent_model_2[st.session_state.agent2_model_choice],
            ],
        },
    }

    asyncio.run(
        websocket_client(
            f"{st.session_state.WS_BASE}/ws/simulation?token=demo-token",
            initial_message,
            timeout=20,
        )
    )


def chat_demo() -> None:
    initialize_session_state()

    with st.sidebar:
        with st.expander("Simulation Setup", expanded=True):
            st.selectbox(
                "Choose a scenario:",
                st.session_state.scenarios.keys(),
                key="scenario_choice",
                disabled=st.session_state.active,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.selectbox(
                    "Choose Agent 1:",
                    st.session_state.agent_list_1.keys(),
                    key="agent_choice_1",
                    disabled=st.session_state.active,
                )
                st.selectbox(
                    "Agent 1 Model:",
                    st.session_state.agent_model_1.keys(),
                    key="agent1_model_choice",
                    disabled=st.session_state.active,
                )

            with col2:
                st.selectbox(
                    "Choose Agent 2:",
                    st.session_state.agent_list_2.keys(),
                    key="agent_choice_2",
                    disabled=st.session_state.active,
                )
                st.selectbox(
                    "Agent 2 Model:",
                    st.session_state.agent_model_2.keys(),
                    key="agent2_model_choice",
                    disabled=st.session_state.active,
                )

        st.button(
            "Start Simulation",
            disabled=st.session_state.active,
            on_click=start_simulation,
        )

    chat_container = st.container()
    while st.session_state.active:
        with chat_container:
            st.write(st.session_state.messages)
        time.sleep(1)

    with chat_container:
        st.write(st.session_state.messages)


chat_demo()
