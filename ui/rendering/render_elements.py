import json

import streamlit as st
from sotopia.database import (
    BaseAgentProfile,
    BaseEnvironmentProfile,
    EpisodeLog,
    BaseCustomEvaluationDimension,
)
from sotopia.envs.parallel import render_text_for_environment

from .render_utils import (
    get_full_name,
    render_messages,
    local_css,
    avatar_mapping,
)
from .get_elements import get_agents


role_mapping = {
    "Background Info": "background",
    "System": "info",
    "Environment": "env",
    "Observation": "obs",
    "General": "eval",
}


def update_database_callback() -> None:
    pass


def display_field(label: str, value: str) -> str:
    if value:
        return f"<p><strong>{label}:</strong> {value}</p>"
    return ""


def render_evaluation_dimension(dimension: BaseCustomEvaluationDimension) -> None:
    local_css("././css/style.css")

    st.markdown(
        f"""
        <div style="background-color: #f9f9f9; padding: 10px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 10px; margin-left: 10px;">
            <p><strong>{dimension.name}:</strong></p>
            <div class="character-truncate">
                <p style="text-overflow: ellipsis; overflow: hidden;">{dimension.description}</p>
                <div style="background-color: #e5dbff ; padding: 10px; border-radius: 10px; margin-bottom: 5px; margin-top: 5px;">
                    <p style="text-overflow: ellipsis; overflow: hidden;"> Range: [{dimension.range_low}, {dimension.range_high}] </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evaluation_dimension_list(
    name: str,
    dimensions: list[BaseCustomEvaluationDimension],
) -> None:
    local_css("././css/style.css")

    all_dimension_names = [dimension.name for dimension in dimensions]

    st.markdown(
        f"""
        <div style="background-color: #f9f9f9; padding: 10px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 10px; margin-left: 10px;">
            <p><strong>{name}:</strong></p>
            <div class="character-truncate">
                <p style="text-overflow: ellipsis; overflow: hidden;">Includes: {', '.join(all_dimension_names)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_character(character: BaseAgentProfile) -> None:
    local_css("././css/style.css")

    full_name = f"{character.first_name} {character.last_name}"
    avatar_file = avatar_mapping.get(full_name, avatar_mapping["default_avatar"])

    # Create two columns: one for the avatar and one for the markdown
    col1, col2 = st.columns([1, 3])  # Adjust the ratio to control column width

    with col1:
        # Display avatar in the first column
        st.image(
            str(avatar_file),
            caption=full_name,
            width=150,  # Set the desired width
        )

    with col2:
        # Display demographic info in the second column
        int_age = getattr(character, "age", 0)
        age = str(int_age) if int_age else ""
        gender = getattr(character, "gender", "")
        occupation = getattr(character, "occupation", "")
        pronouns = getattr(character, "gender_pronoun", "")

        basic_info = [age, gender, pronouns, occupation]

        sub_header = " · ".join(filter(None, basic_info))

        secret = getattr(character, "secret", "")
        secret_header = ""
        if secret:
            secret_header = f"Secret: {secret}"
        st.markdown(
            f"""
            <div style="background-color: #f9f9f9; padding: 10px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 10px; margin-left: 10px;">
                <p><strong>{sub_header}</strong></p>
                <div class="character-truncate">
                    <p style="text-overflow: ellipsis; overflow: hidden;">{getattr(character, "public_info", "")}</p>
                    <div style="background-color: #e5dbff ; padding: 10px; border-radius: 10px; margin-bottom: 5px; margin-top: 5px;">
                        <p style="text-overflow: ellipsis; overflow: hidden;"> {secret_header} </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with st.expander("Personality Info", expanded=False):
        additional_info = ""

        additional_info += display_field(
            "Personality and Values", character.personality_and_values
        )
        additional_info += display_field("Big Five", character.big_five)
        additional_info += display_field(
            "Moral Values", ", ".join(character.moral_values)
        )
        additional_info += display_field(
            "Schwartz Personal Values", ", ".join(character.schwartz_personal_values)
        )
        additional_info += display_field(
            "Decision Making Style", character.decision_making_style
        )
        additional_info += display_field("Model ID", character.model_id)
        additional_info += display_field("MBTI", character.mbti)

        st.markdown(
            f"""
            <div style="background-color: #d0f5d0; padding: 10px; border-radius: 10px; margin-bottom: 12px;">
                {additional_info}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_environment_profile(profile: BaseEnvironmentProfile) -> None:
    # Render the codename as a subheader
    # Render the scenario with domain and realism in styled tags
    if len(profile.agent_goals) == 2:
        processed_agent1_goal = render_text_for_environment(
            profile.agent_goals[0]
        ).replace("\n", "<br>")
        processed_agent2_goal = render_text_for_environment(
            profile.agent_goals[1]
        ).replace("\n", "<br>")
    else:
        processed_agent1_goal = ""
        processed_agent2_goal = ""

    st.markdown(
        f"""
        <div style="background-color: #f9f9f9; padding: 10px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
            <p><strong>Scenario</strong>: {profile.scenario}</p>
            <div style="margin-top: 20px;">
                <div style="display: inline-block; width: 48%; vertical-align: top;">
                    <p><strong>Agent 1's Goal</strong></p>
                    <div style="background-color: #D1E9F6; padding: 10px; border-radius: 10px; margin-bottom: 5px;">
                        <p class="truncate">{processed_agent1_goal}</p>
                    </div>
                </div>
                <div style="display: inline-block; width: 48%; vertical-align: top;">
                    <p><strong>Agent 2's Goal</strong></p>
                    <div style="background-color: #D1E9F6; padding: 10px; border-radius: 10px; margin-bottom: 5px;">
                        <p class="truncate">{processed_agent2_goal}</p>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Foldable green container for additional information
    with st.expander("Additional Information", expanded=False):
        st.write(
            f"""
            <div style="background-color: #d0f5d0; padding: 10px; border-radius: 10px;">
                <h4>Codename</h4>
                <p>{profile.codename}</p>
                <h4>Source</h4>
                <p>{profile.source}</p>
                <h4>Relationship</h4>
                <p>{profile.relationship.name}</p>
                <h4>Age Constraint</h4>
                <p>{profile.age_constraint if profile.age_constraint else 'None'}</p>
                <h4>Occupation Constraint</h4>
                <p>{profile.occupation_constraint if profile.occupation_constraint else 'None'}</p>
                <h4>Agent Constraint</h4>
                <p>{profile.agent_constraint if profile.agent_constraint else 'None'}</p>
                <h4>Tag</h4>
                <p>{profile.tag}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_conversation_and_evaluation(episode: EpisodeLog) -> None:
    local_css("./././css/style.css")
    agents = [list(get_agents(agent).values())[0] for agent in episode.agents]
    agent_names = [get_full_name(agent) for agent in agents]

    messages = render_messages(episode)

    background_messages = [
        message for message in messages if message["role"] == "Background Info"
    ]
    evaluation_messages = [
        message for message in messages if message["type"] == "comment"
    ]
    conversation_messages = [
        message
        for message in messages
        if message not in background_messages and message not in evaluation_messages
    ]

    assert (
        len(background_messages) == 2
    ), f"Need 2 background messages, but got {len(background_messages)}"

    st.markdown("---")

    st.subheader("Conversation & Evaluation")
    with st.expander("Conversation", expanded=True):
        for index, message in enumerate(conversation_messages):
            role = role_mapping.get(message["role"], message["role"])
            content = message["content"]
            # escape doller sign
            content = content.replace("$", "&#36;")

            if role == "obs" or message.get("type") == "action":
                try:
                    content = json.loads(content)
                except Exception as e:
                    print(e)

            with st.chat_message(
                role,
                avatar=str(
                    avatar_mapping.get(message["role"], avatar_mapping["default"])
                ),
            ):
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
                else:
                    if role == agent_names[1]:
                        # add background grey color
                        st.write(f"**{role}**")
                        st.markdown(
                            f"""
                            <div style="background-color: lightgrey; padding: 10px; border-radius: 5px;">
                                {content}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.write(f"**{role}**")
                        st.markdown(
                            content.replace("\n", "<br />"), unsafe_allow_html=True
                        )

    with st.expander("Evaluation Results", expanded=True):
        for message in evaluation_messages:
            st.markdown(
                message["content"].replace("\n", "<br />"), unsafe_allow_html=True
            )
