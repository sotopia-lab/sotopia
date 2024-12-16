from typing import TypedDict, Any
from sotopia.database import EpisodeLog
from pathlib import Path
import streamlit as st


class messageForRendering(TypedDict):
    role: str
    type: str
    content: str


male_links = [
    "https://cmu.box.com/shared/static/a4dx4ys1j5twsxlf5sus4hzmzrlf8fp1.svg",
    "https://cmu.box.com/shared/static/317hzxng0tnnj6osw3bgove5o6ao0wgg.svg",
    "https://cmu.box.com/shared/static/iok10ur3qtsi9dey5jo14151jvvpoppr.svg",
    "https://cmu.box.com/shared/static/317hzxng0tnnj6osw3bgove5o6ao0wgg.svg",
    "https://cmu.box.com/shared/static/ybz7475ouyfdfx6l5rmkz1gfg4118643.svg",
]

female_links = [
    "https://cmu.box.com/shared/static/wahd1spibvl2cxh7nxv3ssa8ev7inqlj.svg",
    "https://cmu.box.com/shared/static/pqyi0rqn8ttvj0ivxgyh9f5pkp03mq6q.svg",
    "https://cmu.box.com/shared/static/0616n35jzpuolzi35236s8kx6ontpiv3.svg",
    "https://cmu.box.com/shared/static/ps2k6j31nowl4z74ospldgxcfiuhgjff.svg",
    "https://cmu.box.com/shared/static/dmonscgihkfm2slcxx2ldtbyil6n1jha.svg",
]

avatar_path = Path("./assets/avatars")
avatar_mapping = {
    "default": "https://cmu.box.com/shared/static/r5xkl977ktt0bke29iuqiafasn900y45.png",
    "default_avatar": "https://cmu.box.com/shared/static/a4dx4ys1j5twsxlf5sus4hzmzrlf8fp1.svg",
    "Samuel Anderson": male_links[0],
    "Zane Bennett": male_links[1],
    "William Brown": male_links[2],
    "Rafael Cortez": male_links[3],
    "Noah Davis": male_links[4],
    "Eli Dawson": male_links[0],
    "Miles Hawkins": male_links[1],
    "Hendrick Heinz": male_links[2],
    "Benjamin Jackson": male_links[3],
    "Ethan Johnson": male_links[4],
    "Liam Johnson": male_links[0],
    "Leo Williams": male_links[1],
    "Finnegan O'Malley": male_links[2],
    "Jaxon Prentice": male_links[3],
    "Donovan Reeves": male_links[4],
    "Micah Stevens": male_links[0],
    "Oliver Thompson": male_links[1],
    "Ethan Smith": male_links[2],
    "Oliver Smith": male_links[3],
    "Baxter Sterling": male_links[4],
    "Jasmine Blake": female_links[0],
    "Sophia Brown": female_links[1],
    "Mia Davis": female_links[2],
    "Naomi Fletcher": female_links[3],
    "Lena Goodwin": female_links[4],
    "Lily Greenberg": female_links[0],
    "Emily Harrison": female_links[1],
    "Amara Hartley": female_links[2],
    "Sophia James": female_links[3],
    "Ava Martinez": female_links[4],
    "Isabelle Martinez": female_links[0],
    "Gwen Pierce": female_links[1],
    "Sasha Ramirez": female_links[2],
    "Giselle Rousseau": female_links[3],
    "Mia Sanders": female_links[4],
    "Calista Sinclair": female_links[0],
    "Esmeralda Solis": female_links[1],
    "Ava Thompson": female_links[2],
    "Imelda Thorne": female_links[3],
    "Isabella White": female_links[4],
}


def get_abstract(description: str) -> str:
    return " ".join(description.split()[:50]) + "..."


def local_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def get_full_name(agent_profile: dict[str, Any]) -> str:
    return f"{agent_profile['first_name']} {agent_profile['last_name']}"


def parse_reasoning(reasoning: str, num_agents: int) -> tuple[list[str], str]:
    """Parse the reasoning string into a dictionary."""
    sep_token = "SEPSEP"
    for i in range(1, num_agents + 1):
        reasoning = (
            reasoning.replace(f"Agent {i} comments:\n", sep_token)
            .strip(" ")
            .strip("\n")
        )
    all_chunks = reasoning.split(sep_token)
    general_comment = all_chunks[0].strip(" ").strip("\n")
    comment_chunks = all_chunks[-num_agents:]

    return comment_chunks, general_comment


def render_messages(episode: EpisodeLog) -> list[messageForRendering]:
    """Generate a list of messages for human-readable version of the episode log."""

    messages_for_rendering: list[messageForRendering] = []

    for idx, turn in enumerate(episode.messages):
        is_observation_printed = False

        if idx == 0:
            assert (
                len(turn) >= 2
            ), "The first turn should have at least environment messages"

            messages_for_rendering.append(
                {"role": "Background Info", "type": "info", "content": turn[0][2]}
            )
            messages_for_rendering.append(
                {"role": "Background Info", "type": "info", "content": turn[1][2]}
            )
            messages_for_rendering.append(
                {"role": "System", "type": "divider", "content": "Start Simulation"}
            )

        for sender, receiver, message in turn:
            if not is_observation_printed and "Observation:" in message and idx != 0:
                extract_observation = message.split("Observation:")[1].strip()
                if extract_observation:
                    messages_for_rendering.append(
                        {
                            "role": "Observation",
                            "type": "observation",
                            "content": extract_observation,
                        }
                    )
                is_observation_printed = True

            if receiver == "Environment":
                if sender != "Environment":
                    if "did nothing" in message:
                        continue
                    elif "left the conversation" in message:
                        messages_for_rendering.append(
                            {
                                "role": "Environment",
                                "type": "leave",
                                "content": f"{sender} left the conversation",
                            }
                        )
                    else:
                        if "said:" in message:
                            message = message.split("said:")[1].strip()
                            messages_for_rendering.append(
                                {"role": sender, "type": "said", "content": message}
                            )
                        else:
                            message = message.replace("[action]", "")
                            messages_for_rendering.append(
                                {"role": sender, "type": "action", "content": message}
                            )
                else:
                    messages_for_rendering.append(
                        {
                            "role": "Environment",
                            "type": "environment",
                            "content": message,
                        }
                    )

    messages_for_rendering.append(
        {"role": "System", "type": "divider", "content": "End Simulation"}
    )

    reasoning_per_agent, general_comment = parse_reasoning(
        episode.reasoning,
        len(
            set(
                msg["role"]
                for msg in messages_for_rendering
                if msg["type"] in {"said", "action"}
            )
        ),
    )

    if general_comment == "":
        return messages_for_rendering[:-1]

    messages_for_rendering.append(
        {"role": "General", "type": "comment", "content": general_comment}
    )

    for idx, reasoning in enumerate(reasoning_per_agent):
        reasoning_lines = reasoning.split("\n")
        new_reasoning = ""
        for reasoning_line in reasoning_lines:
            dimension = reasoning_line.split(":")[0]
            new_reasoning += (
                (f"**{dimension}**: {':'.join(reasoning_line.split(':')[1:])}" + "\n")
                if dimension != ""
                else reasoning_line + "\n"
            )
        messages_for_rendering.append(
            {
                "role": f"Agent {idx + 1}",
                "type": "comment",
                "content": f"**Agent {idx + 1} reasoning**:\n{new_reasoning}\n\n**Rewards**: {str(episode.rewards[idx])}",
            }
        )

    for item in messages_for_rendering:
        item["content"] = item["content"].replace("$", "\\$")

    return messages_for_rendering
