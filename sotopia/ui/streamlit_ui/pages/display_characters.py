import streamlit as st

from ..utils import initialize_session_state

from sotopia.database import AgentProfile


def local_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def display_characters() -> None:
    initialize_session_state()
    st.title("Characters")
    all_characters = AgentProfile.find().all()

    col1, col2 = st.columns(2, gap="medium")
    for i, character in enumerate(all_characters):
        with col1 if i % 2 == 0 else col2:
            assert isinstance(character, AgentProfile)
            rendering_character(character)
            st.write("---")


def display_field(label: str, value: str) -> str:
    if value:
        return f"<p><strong>{label}:</strong> {value}</p>"
    return ""


def rendering_character(character: AgentProfile) -> None:
    local_css("././css/style.css")

    full_name = f"{character.first_name} {character.last_name}"

    personal_info = ""
    personal_info += display_field("Age", str(character.age))
    personal_info += display_field("Occupation", character.occupation)
    personal_info += display_field("Gender", character.gender)
    personal_info += display_field("Gender Pronoun", character.gender_pronoun)

    st.markdown(
        f"""
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
        <p><strong>Name</strong>: {full_name}</p>
        <div style="background-color: #D1E9F6; padding: 10px; border-radius: 10px; margin-bottom: 5px;">
            <p><strong>Demographic Information</strong></p>
            <div class="truncate">
                {personal_info}
            </div>
        </div>

    </div>
    """,
        unsafe_allow_html=True,
    )
    with st.expander("Additional Information", expanded=False):
        additional_info = ""
        additional_info += display_field("Public Info", character.public_info)
        additional_info += display_field("Big Five", character.big_five)
        additional_info += display_field(
            "Moral Values", ", ".join(character.moral_values)
        )
        additional_info += display_field(
            "Schwartz Personal Values", ", ".join(character.schwartz_personal_values)
        )
        additional_info += display_field(
            "Personality and Values", character.personality_and_values
        )
        additional_info += display_field(
            "Decision Making Style", character.decision_making_style
        )
        additional_info += display_field("Secret", character.secret)
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


display_characters()
