import json

import streamlit as st

from sotopia.ui.socialstream.chat.callbacks import (
    other_choice_callback,
    save_callback,
)
from sotopia.ui.socialstream.rendering_utils import (
    compose_agent_messages,
    compose_env_messages,
    messageForRendering,
    render_messages,
)
from sotopia.ui.socialstream.utils import (
    HUMAN_MODEL_NAME,
    ActionState,
    EnvAgentProfileCombo,
    initialize_session_state,
    set_from_env_agent_profile_combo,
    set_settings,
    step,
)


def chat_demo() -> None:
    initialize_session_state()

    def env_agent_choice_callback() -> None:
        if st.session_state.active:
            st.warning("Please stop the conversation first.")
            st.stop()

        agent_choice_1 = st.session_state.agent_choice_1
        agent_choice_2 = st.session_state.agent_choice_2
        if agent_choice_1 == agent_choice_2:
            st.warning(
                "The two agents cannot be the same. Please select different agents."
            )
            st.session_state.error = True
            return

        set_settings(
            agent_choice_1=agent_choice_1,
            agent_choice_2=agent_choice_2,
            scenario_choice=st.session_state.scenario_choice,
            user_agent_name="PLACEHOLDER",
            agent_names=[
                st.session_state.agent_choice_1,
                st.session_state.agent_choice_2,
            ],
            reset_agents=True,
        )

    with st.sidebar:
        with st.expander("Create your scenario!", expanded=True):
            scenarios = st.session_state.env_mapping
            agent_list_1, agent_list_2 = st.session_state.agent_mapping
            target_agents = ["Agent 1", "Agent 2"]

            scenario_col, scenario_desc_col = st.columns(2)
            with scenario_col:
                st.selectbox(
                    "Choose a scenario:",
                    scenarios.keys(),
                    disabled=st.session_state.active,
                    index=0,
                    on_change=env_agent_choice_callback,
                    key="scenario_choice",
                )

            with scenario_desc_col:
                st.markdown(
                    f"""**Scenario Description:** {st.session_state.env_description_mapping[st.session_state.scenario_choice]}""",
                    unsafe_allow_html=True,
                )

            def random_select_callback():
                import random

                agent_choice_1 = random.choice(list(agent_list_1.keys()))
                agent_choice_2 = random.choice(list(agent_list_2.keys()))
                while agent_choice_1 == agent_choice_2:
                    agent_choice_2 = random.choice(list(agent_list_2.keys()))
                st.session_state.agent_choice_1 = agent_choice_1
                st.session_state.agent_choice_2 = agent_choice_2
                env_agent_choice_callback()

                human_agent = random.choice(target_agents)
                st.session_state.human_agent_selection = human_agent
                other_choice_callback(True)

            # when pressing the button, randomly select agents
            st.button(
                "Randomly select agents",
                disabled=st.session_state.active,
                on_click=random_select_callback,
            )

        human_agent_idx = (
            0 if st.session_state.human_agent_selection == "Agent 1" else 1
        )
        agents = st.session_state.agents
        target_agent_viewer = [human_agent_idx + 1 for _ in range(len(agents))]
        agent_infos = compose_agent_messages(
            agents=agents, target_agent_viewer=target_agent_viewer
        )
        env_info, goals_info = compose_env_messages(env=st.session_state.env)
        agent_name = list(agents.keys())[human_agent_idx]

        with st.expander(
            f"You Are **{st.session_state.human_agent_selection}, {agent_name}**",
            expanded=True,
        ):
            st.markdown(
                f"""**Scenario:** {env_info}""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""**Your Background:** {agent_infos[human_agent_idx]}""",
            )
            st.markdown(
                f"""**Your Goal:** {goals_info[human_agent_idx]}""",
            )

        partner_model = st.session_state.agent_models[1 - human_agent_idx]
        with st.expander(f"Your Partner (Model {partner_model}): ", expanded=True):
            target_agent_name = list(agents.keys())[1 - human_agent_idx]
            st.markdown(
                f"""**Your partner is {target_agent_name}. Background:** {agent_infos[1 - human_agent_idx]}""",
            )

        def activate() -> None:
            st.session_state.active = True

        def activate_and_start() -> None:
            activate()

            env_agent_combo = EnvAgentProfileCombo(
                env=st.session_state.env.profile,
                agents=[agent.profile for agent in st.session_state.agents.values()],
            )
            set_from_env_agent_profile_combo(
                env_agent_combo=env_agent_combo, reset_msgs=True
            )

        action_taken: bool = False

        def stop_and_eval() -> None:
            if st.session_state != ActionState.IDLE:
                st.session_state.state = ActionState.EVALUATION_WAITING

        start_col, stop_col, save_col = st.columns(3)
        with start_col:
            start_button = st.button(
                "Start", disabled=st.session_state.active, on_click=activate_and_start
            )
            if start_button:
                # st.session_state.active = True
                st.session_state.state = ActionState.AGENT1_WAITING

        with stop_col:
            stop_button = st.button(
                "Stop", disabled=not st.session_state.active, on_click=stop_and_eval
            )
            if stop_button and st.session_state.active:
                st.session_state.state = ActionState.EVALUATION_WAITING
                with st.spinner("Evaluating..."):
                    step(user_input="")
                    action_taken = True

        with save_col:
            st.download_button(
                label="Save current conversation",
                file_name="saved_conversation.txt",
                mime="text/plain",
                data=save_callback(),
                disabled=st.session_state.active,
                # use_container_width=True
            )

    requires_agent_input = (
        st.session_state.state == ActionState.AGENT1_WAITING
        and st.session_state.agent_models[0] == HUMAN_MODEL_NAME
    ) or (
        st.session_state.state == ActionState.AGENT2_WAITING
        and st.session_state.agent_models[1] == HUMAN_MODEL_NAME
    )

    requires_model_input = (
        st.session_state.state == ActionState.AGENT1_WAITING
        and st.session_state.agent_models[0] != HUMAN_MODEL_NAME
    ) or (
        st.session_state.state == ActionState.AGENT2_WAITING
        and st.session_state.agent_models[1] != HUMAN_MODEL_NAME
    )

    messages = render_messages(
        env=st.session_state.env,
        agent_list=list(st.session_state.agents.values()),
        messages=st.session_state.messages,
        reasoning=st.session_state.reasoning,
        rewards=st.session_state.rewards,
    )
    tag_for_eval = ["Agent 1", "Agent 2", "General"]
    chat_history = [
        message for message in messages if message["role"] not in tag_for_eval
    ]
    evaluation = [message for message in messages if message["role"] in tag_for_eval]

    with st.expander("Chat History", expanded=True):
        streamlit_rendering(chat_history)

    with st.expander("Evaluation"):
        # a small bug: when there is a agent not saying anything there will be no separate evaluation for that agent
        streamlit_rendering(evaluation)

    with st.form("user_input", clear_on_submit=True):
        user_input = st.text_input("Enter your message here:", key="user_input")

        if st.form_submit_button(
            "Submit",
            use_container_width=True,
            disabled=not requires_agent_input,
        ):
            with st.spinner("Agent acting..."):
                st.session_state.state = st.session_state.state + 1
                step(user_input=user_input)
                action_taken = True

    if requires_model_input:
        with st.spinner("Agent acting..."):
            st.session_state.state = st.session_state.state + 1
            step(user_input="")
            action_taken = True

    if st.session_state.state == ActionState.EVALUATION_WAITING:
        print("Evaluating...")
        with st.spinner("Evaluating..."):
            step()

    if action_taken:
        # BUG if the rerun is too fast then the message is not rendering (Seems to be resolved)
        st.rerun()


def streamlit_rendering(messages: list[messageForRendering]) -> None:
    agent1_name, agent2_name = list(st.session_state.agents.keys())[:2]
    avatar_mapping = {
        "env": "🌍",
        "obs": "🌍",
    }

    agent_names = [agent1_name, agent2_name]
    avatar_mapping = {
        agent_name: "👤"
        if st.session_state.agent_models[idx] == HUMAN_MODEL_NAME
        else "🤖"
        for idx, agent_name in enumerate(agent_names)
    }  # TODO maybe change the avatar because all bot/human will cause confusion

    role_mapping = {
        "Background Info": "background",
        "System": "info",
        "Environment": "env",
        "Observation": "obs",
        "General": "eval",
        "Agent 1": agent1_name,
        "Agent 2": agent2_name,
        agent1_name: agent1_name,
        agent2_name: agent2_name,
    }

    for index, message in enumerate(messages):
        role = role_mapping.get(message["role"], "info")
        content = message["content"]

        if role == "background":
            continue

        if role == "obs" or message.get("type") == "action":
            try:
                content = json.loads(content)
            except Exception as e:
                print(e)

        with st.chat_message(role, avatar=avatar_mapping.get(role, None)):
            if isinstance(content, dict):
                st.json(content)
            elif role == "info":
                st.markdown(
                    f"""
                    <div style="background-color: lightblue; padding: 10px; border-radius: 5px;">
                        {content}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif role in [agent1_name, agent2_name]:
                st.write(f"**{role}**")
                st.markdown(content.replace("\n", "<br />"), unsafe_allow_html=True)
            else:
                st.markdown(content.replace("\n", "<br />"), unsafe_allow_html=True)
