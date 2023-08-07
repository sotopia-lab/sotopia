import json
import logging

import pandas as pd
from rich.logging import RichHandler

from sotopia.agents import (
    generate_background,
    generate_background_conversation,
)
from sotopia.messages import AgentAction
from sotopia.server import run_sync_server

FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

logging.basicConfig(
    level=15,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(),
    ],
)

basic_info_df = pd.read_csv("./data/character_basic_info.csv")
choosen_character = "Ava Martinez"
background_json_file = (
    f"data/background_{'_'.join(choosen_character.lower().split(' '))}.json"
)
info_json_file = (
    f"data/info_{'_'.join(choosen_character.lower().split(' '))}.json"
)
row_num = basic_info_df[basic_info_df["name"] == choosen_character].index[0]
basic_info = basic_info_df.loc[row_num].to_dict()

(
    initial_profile,
    profile,
    first_narrative,
    second_narrative,
    previous_messages,
) = generate_background(info_json_file, basic_info)
conversation_seeds = pd.read_csv("./data/conversation_seeds.csv")
for index, row in conversation_seeds.iterrows():
    seed = row.to_dict()
    messages, background = generate_background_conversation(
        seed,
        basic_info,
        initial_profile,
        profile,
        background_json_file,
        run_sync_server,
    )
    info_box = {
        "initial_profile": basic_info,
        "profile": profile,
        "second_narrative_profile": second_narrative,
        "first_narrative_profile": first_narrative,
        "overall_background": background.dict(),
        "messages": previous_messages,
    }
    assert isinstance(info_box["messages"], list)
    for index, (sender, receiver, message) in enumerate(messages):
        if receiver == "Environment":
            assert isinstance(message, AgentAction)
            if message.action_type != "none":
                msg_dict = message.dict()
                msg_dict["agent_name"] = sender
                msg_dict["scenario"] = background.scenario
                msg_dict["questioner_goal"] = background.p1_goal
                info_box["messages"].append(msg_dict)
    with open(info_json_file, "w") as f:
        json.dump(info_box, f, indent=4)
