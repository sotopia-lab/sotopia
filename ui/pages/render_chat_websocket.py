import asyncio
import json
import threading
import time
from queue import Queue
from typing import Any, Optional

import aiohttp
import streamlit as st

from sotopia.database import BaseEpisodeLog, BaseEnvironmentProfile
from ui.rendering import (
    get_scenarios,
    get_agents,
    get_models,
    get_evaluation_dimensions,
    render_environment_profile,
    get_abstract,
    render_conversation_and_evaluation,
)


def initialize_session_state() -> None:
    if "active" not in st.session_state:
        # Initialize base state
        st.session_state.scenarios = get_scenarios()
        st.session_state.agent_dict = get_agents()
        st.session_state.agent_model_dict = get_models()
        st.session_state.evaluation_dimension_dict = get_evaluation_dimensions()

        st.session_state.scenarios = dict(
            sorted(
                st.session_state.scenarios.items(),
                key=lambda item: item[0],
                reverse=True,
            )
        )
        st.session_state.agent_dict = dict(
            sorted(
                st.session_state.agent_dict.items(),
                key=lambda item: item[0],
            )
        )

        # Use first items as default choices
        st.session_state.scenario_choice = list(st.session_state.scenarios.keys())[0]
        st.session_state.agent_choice_1 = list(st.session_state.agent_dict.keys())[0]
        st.session_state.agent_choice_2 = list(st.session_state.agent_dict.keys())[0]
        st.session_state.agent1_model_choice = list(
            st.session_state.agent_model_dict.keys()
        )[0]
        st.session_state.agent2_model_choice = list(
            st.session_state.agent_model_dict.keys()
        )[0]
        st.session_state.evaluation_dimension_choice = list(
            st.session_state.evaluation_dimension_dict.keys()
        )[0]

        # Initialize websocket manager and message list
        st.session_state.messages = []
        # Set initial active state
        st.session_state.active = False

        st.session_state.websocket_manager = WebSocketManager(
            f"{st.session_state.WS_BASE}/ws/simulation?token=demo-token"
        )
        print("Session state initialized")


chat_history_container = st.empty()


class WebSocketManager:
    def __init__(self, url: str):
        self.url = url
        self.websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        self.message_queue: Queue[str | dict[str, Any]] = Queue()
        self.running: bool = False
        self.receive_queue: Queue[dict[str, Any]] = Queue()
        self._closed = threading.Event()

    def start(self) -> None:
        """Start the client in a separate thread"""
        self._closed.clear()
        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop)
        self.thread.start()

    def stop(self) -> None:
        """Stop the client"""
        print("Stopping websocket manager...")
        self.running = False
        self._closed.wait(timeout=5.0)
        if self.thread.is_alive():
            print("Thread is still alive after stop")
        else:
            print("Thread has been closed")

    def send_message(self, message: str | dict[str, Any]) -> None:
        """Add a message to the queue to be sent"""
        if isinstance(message, dict):
            message = json.dumps(message)
        self.message_queue.put(message)

    def _run_event_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._connect())

    async def _connect(self) -> None:
        """Connect to the WebSocket server and handle messages"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(self.url) as ws:
                    self.websocket = ws

                    # Start tasks for sending and receiving messages
                    send_task = asyncio.create_task(self._send_messages())
                    receive_task = asyncio.create_task(self._receive_messages())

                    # Wait for both tasks to complete
                    try:
                        await asyncio.gather(send_task, receive_task)
                    except Exception as e:
                        print(f"Error in tasks: {e}")
                    finally:
                        send_task.cancel()
                        receive_task.cancel()
        finally:
            print("WebSocket connection closed")
            self._closed.set()

    async def _send_messages(self) -> None:
        """Send messages from the queue"""
        while self.running:
            if not self.message_queue.empty():
                message = self.message_queue.get()
                await self.websocket.send_str(message)  # type: ignore
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

    async def _receive_messages(self) -> None:
        """Receive and handle incoming messages"""
        while self.running:
            try:
                msg = await self.websocket.receive()  # type: ignore
                if msg.type == aiohttp.WSMsgType.TEXT:
                    print(f"Received message: {msg.data}")
                    self.receive_queue.put(json.loads(msg.data))
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
            except Exception as e:
                print(f"Error receiving message: {e}")
                break


def set_active(value: bool) -> None:
    st.session_state.active = value


def handle_end(message: dict[str, Any]) -> None:
    set_active(False)
    st.session_state.websocket_manager.stop()


def handle_error_msg(message: dict[str, Any]) -> None:
    # TODO handle different error
    print("[!!] Error in message: ", message)
    st.error(f"Error in message: {message['data']['content']}")


def handle_server_msg(message: dict[str, Any]) -> None:
    msg_type = message["data"]["type"]
    if msg_type == "messages":
        epilog = BaseEpisodeLog(**message["data"]["messages"])
        st.session_state.messages.append(epilog)


def handle_message(message: dict[str, Any]) -> None:
    # "END_SIM", "SERVER_MSG", "ERROR",
    match message["type"]:
        case "END_SIM":
            st.session_state.websocket_manager.stop()
            st.rerun()
        case "SERVER_MSG":
            handle_server_msg(message)
        case "ERROR":
            handle_error_msg(message)
        case _:
            st.error(f"Unknown message type: {message['data']['type']}")


def copy_current_json_config() -> None:
    """Copy the current JSON configuration to the text area."""
    config = generate_current_json_config()
    st.session_state.json_config = config
    st.session_state.sidebar_json_config = config


def start_from_json_config() -> None:
    """Start a simulation using a JSON configuration."""
    try:
        # Determine which JSON config to use (main or sidebar)
        if st.session_state.get("_is_sidebar", False):
            json_config = st.session_state.sidebar_json_config
        else:
            json_config = st.session_state.json_config

        # Parse the JSON configuration
        config = json.loads(json_config)

        # Validate the configuration
        if "type" not in config or "data" not in config:
            st.error("Invalid configuration: missing 'type' or 'data' field")
            return

        # Start the simulation
        st.session_state.active = True
        st.session_state.messages = []
        chat_history_container.empty()
        st.session_state.websocket_manager.start()
        st.session_state.websocket_manager.send_message(config)

    except json.JSONDecodeError:
        st.error("Invalid JSON configuration. Please check your syntax.")
    except Exception as e:
        st.error(f"Error starting simulation: {str(e)}")


def start_from_sidebar_json() -> None:
    """This function is no longer used."""
    pass


def start_from_main_json() -> None:
    """Start a simulation using the main JSON configuration."""
    st.session_state._is_sidebar = False
    start_from_json_config()


def generate_default_json_config() -> str:
    """Generate a default JSON configuration."""
    # Get the first scenario and agent
    scenario_key = list(st.session_state.scenarios.keys())[0]
    agent_key = list(st.session_state.agent_dict.keys())[0]
    model_key = list(st.session_state.agent_model_dict.keys())[0]
    eval_key = list(st.session_state.evaluation_dimension_dict.keys())[0]

    # Create a default configuration
    config = {
        "type": "START_SIM",
        "data": {
            "env_id": st.session_state.scenarios[scenario_key]["pk"],
            "agent_ids": [
                st.session_state.agent_dict[agent_key]["pk"],
                st.session_state.agent_dict[agent_key]["pk"],
            ],
            "agent_models": [
                st.session_state.agent_model_dict[model_key],
                st.session_state.agent_model_dict[model_key],
            ],
            "evaluator_model": "gpt-4o",
            "evaluation_dimension_list_name": eval_key,
            "max_turns": 20,
        },
    }

    config = {
        "type": "START_SIM",
        "data": {
            "env_id": "env_123",
            "agent_ids": ["agent_1", "agent_2", "agent_3"],
            "agent_models": ["gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"],
            "evaluator_model": "gpt-4o",
            "evaluation_dimension_list_name": "sotopia",
            "env_profile_dict": {
                "codename": "test",
                "scenario": "Just chat (finish the conversation in 2 turns)",
                "agent_goals": ["Just chat", "Just chat", "Just chat"],
            },
            "agent_profile_dicts": [
                {"first_name": "agent_1", "last_name": "agent_1"},
                {"first_name": "agent_2", "last_name": "agent_2"},
                {"first_name": "agent_3", "last_name": "agent_3"},
            ],
            "max_turns": 20,
        },
    }
    return json.dumps(config, indent=2)


def generate_current_json_config() -> str:
    """Generate a JSON configuration based on the current settings."""
    config = {
        "type": "START_SIM",
        "data": {
            "env_id": st.session_state.scenarios[st.session_state.scenario_choice][
                "pk"
            ],
            "agent_ids": [
                st.session_state.agent_dict[st.session_state.agent_choice_1]["pk"],
                st.session_state.agent_dict[st.session_state.agent_choice_2]["pk"],
            ],
            "agent_models": [
                st.session_state.agent_model_dict[
                    st.session_state.agent_model_choice_1
                ],
                st.session_state.agent_model_dict[
                    st.session_state.agent_model_choice_2
                ],
            ],
            "evaluator_model": "gpt-4o",
            "evaluation_dimension_list_name": st.session_state.evaluation_dimension_choice,
            "max_turns": int(st.session_state.max_turns),
        },
    }

    return json.dumps(config, indent=2)


def start_callback() -> None:
    if st.session_state.agent_choice_1 == st.session_state.agent_choice_2:
        st.error("Please select different agents")
    else:
        st.session_state.active = True
        st.session_state.messages = []
        chat_history_container.empty()
        st.session_state.websocket_manager.start()

        # Create the configuration in the same format as the JSON config
        config = {
            "type": "START_SIM",
            "data": {
                "env_id": st.session_state.scenarios[st.session_state.scenario_choice][
                    "pk"
                ],
                "agent_ids": [
                    st.session_state.agent_dict[st.session_state.agent_choice_1]["pk"],
                    st.session_state.agent_dict[st.session_state.agent_choice_2]["pk"],
                ],
                "agent_models": [
                    st.session_state.agent_model_dict[
                        st.session_state.agent_model_choice_1
                    ],
                    st.session_state.agent_model_dict[
                        st.session_state.agent_model_choice_2
                    ],
                ],
                "evaluator_model": "gpt-4o",
                "evaluation_dimension_list_name": st.session_state.evaluation_dimension_choice,
                "max_turns": int(st.session_state.max_turns),
            },
        }

        st.session_state.websocket_manager.send_message(config)


def stop_callback() -> None:
    st.session_state.stop_sim = True
    st.session_state.websocket_manager.send_message(
        {
            "type": "FINISH_SIM",
            "data": "",
        }
    )


def update_scenario_description() -> None:
    scenario = st.session_state.scenarios[st.session_state.scenario_choice]
    environment_profile = BaseEnvironmentProfile(**scenario)
    render_environment_profile(environment_profile)


def is_active() -> bool:
    active_state = st.session_state.websocket_manager.running
    assert isinstance(active_state, bool)
    return active_state


def chat_demo() -> None:
    initialize_session_state()

    # Add a checkbox to toggle between JSON configuration and standard UI
    use_json_config = st.checkbox("Use JSON Configuration", value=False)

    if use_json_config:
        # Only show JSON configuration when checkbox is selected
        st.markdown("""
        ### Configure and start a simulation with JSON

        Enter your JSON configuration below and click "Start Simulation" to begin.
        """)

        col1, col2 = st.columns([3, 1])
        with col2:
            st.button("Use Current Settings", on_click=copy_current_json_config)

        # Initialize the JSON config if it doesn't exist
        if "json_config" not in st.session_state:
            st.session_state.json_config = generate_default_json_config()

        st.text_area(
            "JSON Configuration:",
            st.session_state.json_config,
            height=300,
            key="json_config",
        )

        if st.button(
            "Start Simulation from JSON",
            key="json_start_button",
            on_click=start_from_main_json,
        ):
            pass

        st.markdown("""
        **Example Configuration:**
        ```json
        {
          "type": "START_SIM",
          "data": {
            "env_id": "env_123",
            "agent_ids": ["agent_1", "agent_2", "agent_3"],
            "agent_models": ["gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"],
            "evaluator_model": "gpt-4o",
            "evaluation_dimension_list_name": "sotopia",
            "env_profile_dict": {
              "codename": "test",
              "scenario": "Just chat (finish the conversation in 2 turns)",
              "agent_goals": ["Just chat", "Just chat", "Just chat"]
            },
            "agent_profile_dicts": [
              {"first_name": "agent_1", "last_name": "agent_1"},
              {"first_name": "agent_2", "last_name": "agent_2"},
              {"first_name": "agent_3", "last_name": "agent_3"}
            ],
            "max_turns": 20
          }
        }
        ```
        """)
    else:
        # Show standard UI with scenario description when JSON config is not selected
        update_scenario_description()

        # Remove the JSON expander from standard UI view

    with st.sidebar:
        with st.container():
            # Remove JSON Configuration from sidebar

            # Scenario and Agent Selection
            with st.expander("Simulation Setup", expanded=True):
                scenario_col, scenario_desc_col = st.columns(2)
                with scenario_col:
                    st.selectbox(
                        "Choose a scenario:",
                        st.session_state.scenarios.keys(),
                        key="scenario_choice",
                        disabled=is_active() or use_json_config,
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
                        list(st.session_state.agent_dict.keys()),
                        key="agent_choice_1",
                        disabled=is_active() or use_json_config,
                    )

                with agent2_col:
                    st.selectbox(
                        "Choose Agent 2:",
                        list(st.session_state.agent_dict.keys()),
                        key="agent_choice_2",
                        disabled=is_active() or use_json_config,
                    )

                model1_col, model2_col = st.columns(2)
                with model1_col:
                    st.selectbox(
                        "Choose Agent 1 Model:",
                        list(st.session_state.agent_model_dict.keys()),
                        key="agent_model_choice_1",
                        disabled=is_active() or use_json_config,
                    )

                with model2_col:
                    st.selectbox(
                        "Choose Agent 2 Model:",
                        list(st.session_state.agent_model_dict.keys()),
                        key="agent_model_choice_2",
                        disabled=is_active() or use_json_config,
                    )

                st.selectbox(
                    "Choose evaluation dimensions:",
                    list(st.session_state.evaluation_dimension_dict.keys()),
                    key="evaluation_dimension_choice",
                    disabled=is_active() or use_json_config,
                )

                evaluation_dimension_str = f"**Evaluation Dimensions:** {st.session_state.evaluation_dimension_choice}. <br>**Metric includes:** "
                for eval_dim in st.session_state.evaluation_dimension_dict[
                    st.session_state.evaluation_dimension_choice
                ]:
                    evaluation_dimension_str += f"{eval_dim.name}, "

                st.markdown(
                    evaluation_dimension_str[:-2] + ".",
                    unsafe_allow_html=True,
                )

        with st.expander("Other Options", expanded=False):
            st.text_input(
                "Max Turns", key="max_turns", value="20", disabled=use_json_config
            )
            st.text_input(
                "Max Stale Turns",
                key="max_stale_turns",
                value="3",
                disabled=use_json_config,
            )

        # Control Buttons
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            st.button(
                "Start Simulation",
                disabled=is_active() or use_json_config,
                on_click=start_callback,
            )

        with col2:
            st.button(
                "Stop Simulation",
                disabled=not is_active(),
                on_click=stop_callback,
            )

    chat_history_container = st.empty()
    while is_active():
        if (
            "websocket_manager" in st.session_state
            and st.session_state.websocket_manager.receive_queue.qsize() > 0
        ):
            # get messages one by one and process them

            while not st.session_state.websocket_manager.receive_queue.empty():
                message = st.session_state.websocket_manager.receive_queue.get()
                handle_message(message)

        with chat_history_container.container():
            if st.session_state.messages:
                render_conversation_and_evaluation(st.session_state.messages[-1])
        time.sleep(1)

    with chat_history_container.container():
        if st.session_state.messages:
            render_conversation_and_evaluation(st.session_state.messages[-1])


chat_demo()
