import json

import streamlit as st

from sotopia.ui.socialstream.chat.callbacks import (
    agent_edit_callback_finegrained,
    edit_callback,
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
    MODEL_LIST,
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
        st.checkbox(
            "Make the scenario editable",
            key="edit_scenario",
            on_change=other_choice_callback,
            disabled=st.session_state.active,
        )

        with st.expander("Create your scenario!", expanded=True):
            scenarios = st.session_state.env_mapping
            agent_list_1, agent_list_2 = st.session_state.agent_mapping

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

            agent_col1, agent_col2 = st.columns(2)
            with agent_col1:
                agent_choice_1 = st.selectbox(
                    "Choose Agent 1:",
                    agent_list_1.keys(),
                    disabled=st.session_state.active,
                    index=0,
                    on_change=env_agent_choice_callback,
                    key="agent_choice_1",
                )
            with agent_col2:
                agent_choice_2 = st.selectbox(
                    "Choose Agent 2:",
                    agent_list_2.keys(),
                    disabled=st.session_state.active,
                    index=1,
                    on_change=env_agent_choice_callback,
                    key="agent_choice_2",
                )
            if agent_choice_1 == agent_choice_2:
                st.warning(
                    "The two agents cannot be the same. Please select different agents."
                )
                st.stop()

            model_col_1, model_col_2 = st.columns(2)
            with model_col_1:
                st.selectbox(
                    "Choose a model:",
                    MODEL_LIST,
                    disabled=st.session_state.active,
                    index=0,
                    on_change=other_choice_callback,
                    key="agent1_model_choice",
                )
            with model_col_2:
                st.selectbox(
                    "Choose a model for agent 2:",
                    MODEL_LIST,
                    disabled=st.session_state.active,
                    index=0,
                    on_change=other_choice_callback,
                    key="agent2_model_choice",
                )

        with st.expander("Check your social task!", expanded=True):
            agent_infos = compose_agent_messages(agents=st.session_state.agents)
            env_info, goals_info = compose_env_messages(env=st.session_state.env)

            if st.session_state.editable:
                st.text_area(
                    label="Change the scenario here:",
                    value=f"""{env_info}""",
                    height=50,
                    on_change=edit_callback,
                    key="edited_scenario",
                    disabled=st.session_state.active or not st.session_state.editable,
                    args=("edited_scenario",),
                )

                # agent: first name, last name, age, occupation, public info, personality and values, secret
                # use separate text_area for each info
                agent_list = list(st.session_state.agents.values())
                agent_traits = [
                    "first_name",
                    "last_name",
                    "age",
                    "occupation",
                    "personality_and_values",
                    "public_info",
                    "secret",
                ]
                agent_traits = [
                    ["first_name", "last_name", "age"],
                    ["occupation", "personality_and_values"],
                    ["public_info", "secret"],
                ]

                for agent_idx, agent in enumerate(agent_list):
                    st.markdown(f"**Agent {agent_idx + 1} information**")
                    # basic_info_cols = st.columns([1, 1, 1, 1, 3, 3, 3])
                    for trait_set in agent_traits:
                        basic_info_cols = st.columns(len(trait_set))
                        for idx, (trait_name, trait_col) in enumerate(
                            zip(trait_set, basic_info_cols)
                        ):
                            with trait_col:
                                st.text_area(
                                    label=f"{trait_name.capitalize()}",
                                    value=f"""{getattr(agent_list[agent_idx].profile, trait_name)}""",
                                    height=5,
                                    disabled=st.session_state.active
                                    or not st.session_state.editable,
                                    on_change=agent_edit_callback_finegrained,
                                    key=f"edited_agent-{agent_idx}-{trait_name}",
                                    args=(
                                        f"edited_agent-{agent_idx}-{trait_name}",
                                        agent_traits,
                                    ),
                                )

                st.markdown("Goals")
                agent1_goal_col, agent2_goal_col = st.columns(2)
                agent_goal_cols = [agent1_goal_col, agent2_goal_col]
                for agent_idx, goal_info in enumerate(goals_info):
                    agent_goal_col = agent_goal_cols[agent_idx]
                    with agent_goal_col:
                        st.text_area(
                            label=f"Change the goal for Agent {agent_idx + 1} here:",
                            value=f"""{goal_info}""",
                            height=150,
                            key=f"edited_goal_{agent_idx}",
                            on_change=edit_callback,
                            disabled=st.session_state.active
                            or not st.session_state.editable,
                            args=(f"edited_goal_{agent_idx}",),
                        )
            else:
                st.markdown(
                    f"""**Scenario:** {env_info}""",
                    unsafe_allow_html=True,
                )

                agent1_col, agent2_col = st.columns(2)
                agent_cols = [agent1_col, agent2_col]
                for agent_idx, agent_info in enumerate(agent_infos):
                    agent_col = agent_cols[agent_idx]
                    with agent_col:
                        st.markdown(
                            f"""**Agent {agent_idx + 1} Background:** {agent_info}""",
                            unsafe_allow_html=True,
                        )

                agent1_goal_col, agent2_goal_col = st.columns(2)
                agent_goal_cols = [agent1_goal_col, agent2_goal_col]
                for agent_idx, goal_info in enumerate(goals_info):
                    agent_goal_col = agent_goal_cols[agent_idx]
                    with agent_goal_col:
                        st.markdown(
                            f"""**Agent {agent_idx + 1} Goal:** {goal_info}""",
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
        # time.sleep(0.5)  # sleep for a while to prevent running too fast
        # BUG if the rerun is too fast then the message is not rendering (seems to be resolved)
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
