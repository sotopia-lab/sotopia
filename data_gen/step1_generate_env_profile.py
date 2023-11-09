import asyncio
import random
from typing import TypeVar
from tqdm import tqdm

import pandas as pd
import rich
from pydantic import BaseModel

from sotopia.database import EnvironmentProfile
from generate import agenerate_env_profile

T = TypeVar("T", bound=BaseModel)

def pydantics_to_csv(filename: str, data: list[T]) -> None:
    pd.DataFrame([item.dict() for item in data]).to_csv(filename, index=False)


#random.seed(41)

envs = EnvironmentProfile.find().all()
ins_prompts = pd.read_csv("./inspirational_prompt_for_env.csv")
prompts = [prompt.strip().replace('\"', '') for prompt in ins_prompts["prompt"].tolist()]

# randomly choose 3 prompts
sampled_examples = []
sampled_prompts = []

target_num = 500

for i in range(target_num):
    sampled_envs = random.sample(envs, 5)
    sampled_prompt = random.sample(prompts, 5)
    sampled_examples.append(f"1.{sampled_envs[0].json()}\n2.{sampled_envs[1].json()}\n3.{sampled_envs[2].json()}\n4.{sampled_envs[3].json()}\n5.{sampled_envs[4].json()}")
    sampled_prompts.append(f"1.{sampled_prompt[0]}\n2.{sampled_prompt[1]}\n3.{sampled_prompt[2]}\n4.{sampled_prompt[3]}\n5.{sampled_prompt[4]}")

assert len(sampled_examples) == target_num
assert len(sampled_prompts) == target_num

backgrounds = []
for prompt, sampled_example in tqdm(zip(sampled_prompts, sampled_examples), total=target_num):
    rich.print(prompt)
    try:
        background, prompt_full = asyncio.run(
            agenerate_env_profile(
                model_name="gpt-4-turbo",
                inspiration_prompt=prompt,
                examples=sampled_example,
                temperature=0.5,
            )
        )
    except Exception as e:
        print(e)
        print('error! Skip')
        continue
    rich.print(prompt_full)
    rich.print(background)
    backgrounds.append(background)

    pydantics_to_csv("./backgrounds_gpt-4-turbo_jason.csv", backgrounds)