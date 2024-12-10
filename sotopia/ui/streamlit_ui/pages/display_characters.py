import streamlit as st

from sotopia.database import AgentProfile

# Importing avatars
from pathlib import Path

avatar_path = Path('./assets/avatars')
avatars = { 
    'Samuel Anderson': avatar_path / 'male/avatar-svgrepo-com_1_blue.svg',
    'Zane Bennett': avatar_path / 'male/avatar-svgrepo-com_2_blue.svg',
    'William Brown': avatar_path / 'male/avatar-svgrepo-com_3_blue.svg',
    'Rafael Cortez': avatar_path / 'male/avatar-svgrepo-com_4_blue.svg',
    'Noah Davis': avatar_path / 'male/avatar-svgrepo-com_5_blue.svg',
    'Eli Dawson': avatar_path / 'male/avatar-svgrepo-com_6_blue.svg',
    'Miles Hawkins': avatar_path / 'male/avatar-svgrepo-com_7_blue.svg',
    'Hendrick Heinz': avatar_path / 'male/avatar-svgrepo-com_8_blue.svg',
    'Benjamin Jackson': avatar_path / 'male/avatar-svgrepo-com_1_green.svg',
    'Ethan Johnson': avatar_path / 'male/avatar-svgrepo-com_2_green.svg',
    'Liam Johnson': avatar_path / 'male/avatar-svgrepo-com_3_green.svg',
    'Leo Williams': avatar_path / 'male/avatar-svgrepo-com_4_green.svg',
    "Finnegan O'Malley": avatar_path / 'male/avatar-svgrepo-com_4_purple.svg',
    'Jaxon Prentice': avatar_path / 'male/avatar-svgrepo-com_5_green.svg',
    'Donovan Reeves': avatar_path / 'male/avatar-svgrepo-com_6_green.svg',
    'Micah Stevens': avatar_path / 'male/avatar-svgrepo-com_7_green.svg',
    'Oliver Thompson': avatar_path / 'male/avatar-svgrepo-com_8_green.svg',
    'Ethan Smith': avatar_path / 'male/avatar-svgrepo-com_1_purple.svg',
    'Oliver Smith': avatar_path / 'male/avatar-svgrepo-com_2_purple.svg',
    'Baxter Sterling': avatar_path / 'male/avatar-svgrepo-com_3_purple.svg',
    'Jasmine Blake': avatar_path / 'female/avatar-svgrepo-com_10_green.svg',
    'Sophia Brown': avatar_path / 'female/avatar-svgrepo-com_1_blue.svg',
    'Mia Davis': avatar_path / 'female/avatar-svgrepo-com_2_blue.svg',
    'Naomi Fletcher': avatar_path / 'female/avatar-svgrepo-com_3_blue.svg',
    'Lena Goodwin': avatar_path / 'female/avatar-svgrepo-com_4_blue.svg',
    'Lily Greenberg': avatar_path / 'female/avatar-svgrepo-com_5_blue.svg',
    'Emily Harrison': avatar_path / 'female/avatar-svgrepo-com_6_blue.svg',
    'Amara Hartley': avatar_path / 'female/avatar-svgrepo-com_7_blue.svg',
    'Sophia James': avatar_path / 'female/avatar-svgrepo-com_8_blue.svg',
    'Ava Martinez': avatar_path / 'female/avatar-svgrepo-com_9_blue.svg',
    'Isabelle Martinez': avatar_path / 'female/avatar-svgrepo-com_10_blue.svg',
    'Gwen Pierce': avatar_path / 'female/avatar-svgrepo-com_1_green.svg',
    'Sasha Ramirez': avatar_path / 'female/avatar-svgrepo-com_2_green.svg',
    'Giselle Rousseau': avatar_path / 'female/avatar-svgrepo-com_3_green.svg',
    'Mia Sanders': avatar_path / 'female/avatar-svgrepo-com_4_green.svg',
    'Calista Sinclair': avatar_path / 'female/avatar-svgrepo-com_5_green.svg',
    'Esmeralda Solis': avatar_path / 'female/avatar-svgrepo-com_6_green.svg',
    'Ava Thompson': avatar_path / 'female/avatar-svgrepo-com_7_green.svg',
    'Imelda Thorne': avatar_path / 'female/avatar-svgrepo-com_8_green.svg',
    'Isabella White': avatar_path / 'female/avatar-svgrepo-com_9_green.svg',
}
  
def local_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def display_characters() -> None:
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
    avatar_file = avatars.get(full_name, avatars['Samuel Anderson'])

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

        
        st.markdown(
            f"""
            <div style="background-color: #f9f9f9; padding: 10px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
                <p><strong>{sub_header}</strong></p>
                <div class="character-truncate">
                    <p style="text-overflow: ellipsis; overflow-hidden;"> {display_field("Public Info", character.public_info)} </p>
                    <div style="background-color: #FFB6B6 ; padding: 10px; border-radius: 10px; margin-bottom: 5px;">
                        {display_field("Secret", character.secret)}
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


display_characters()
