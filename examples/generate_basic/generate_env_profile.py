import asyncio
import random
from typing import TypeVar

import pandas as pd
import rich
from pydantic import BaseModel

from sotopia.database import EnvironmentProfile
from sotopia.generation_utils.generate import agenerate_env_profile

random.seed(41)

env_borrowMoney = EnvironmentProfile.find(
    EnvironmentProfile.codename == "borrow_money"
).all()[0]
env_roadtrip = EnvironmentProfile.find(
    EnvironmentProfile.codename == "take_turns"
).all()[0]
env_prisonerDillema = EnvironmentProfile.find(
    EnvironmentProfile.codename == "prison_dilemma"
).all()[0]

examples = f"{env_borrowMoney.json()}\n\n{env_roadtrip.json()}\n\n{env_prisonerDillema.json()}"

ins_prompts = pd.read_csv("data/inspirational_prompt_for_env.csv")
prompts = ins_prompts["prompt"].tolist()

T = TypeVar("T", bound=BaseModel)


def pydantics_to_csv(filename: str, data: list[T]) -> None:
    pd.DataFrame([item.dict() for item in data]).to_csv(filename, index=False)


backgrounds = []
for prompt in prompts:
    rich.print(prompt)
    background, prompt_full = asyncio.run(
        agenerate_env_profile(
            model_name="gpt-4",
            inspiration_prompt=prompt,
            examples=examples,
        )
    )
    rich.print(background)
    rich.print(prompt_full)
    backgrounds.append(background)

pydantics_to_csv("data/backgrounds.csv", backgrounds)
