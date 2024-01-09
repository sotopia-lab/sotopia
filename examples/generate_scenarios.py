import typer
from sotopia.database import EnvironmentProfile
from typing import Any, cast
from sotopia.generation_utils import LLM_Name, generate_mutual_friend_envs, generate_craigslist_bargains_envs
from experiment_eval import _sample_env_agent_combo_and_push_to_db
from redis_om import Migrator
import asyncio
import ast
import pandas as pd

app = typer.Typer()

def add_env_profile(**kwargs: dict[str, Any]) -> EnvironmentProfile:
    env_profile = EnvironmentProfile(**kwargs)
    env_profile.save()
    return env_profile


def add_env_profiles(env_profiles: list[dict[str, Any]]) -> list[EnvironmentProfile]:
    env_list = []
    for env_profile in env_profiles:
        profile = add_env_profile(**env_profile)
        env_list.append(profile)
    return env_list


def check_existing_envs(env_profile: dict[str, Any], existing_envs: list[EnvironmentProfile]) -> bool:
    for env in existing_envs:
        if env_profile["scenario"] == env.scenario and env_profile["agent_goals"] == env.agent_goals:
            return False
    return True

def generate_newenv_profile(num: int, gen_model: LLM_Name="gpt-4-turbo", temperature: float=0.5, type: str='mutual_friend') -> pd.DataFrame:
    env_profile_list = [] # type: ignore
    existing_envs = [EnvironmentProfile.get(pk) for pk in EnvironmentProfile.all_pks()]
    if type == "mutual_friend":
        while len(env_profile_list) < num:
            scenario, social_goals = asyncio.run(generate_mutual_friend_envs())
            env_profile = {
                "codename": "mutual_friend",
                "scenario": scenario,
                "agent_goals": social_goals,
                "relationship": 0,
                "age_constraint": 0,
                "occupation_constraint": 0,
                "source": "generated",
            }
            if check_existing_envs(env_profile, existing_envs):
                env_profile_list.append(env_profile)
    else:
        raise NotImplementedError("Only mutual_friend is supported for now")
    return pd.DataFrame(env_profile_list)

@app.command()
def auto_generate_scenarios(num: int, gen_model: LLM_Name="gpt-4-turbo", temperature: float=0.5) -> None:
    """
    Function to generate new environment scenarios based on target number of generation
    """
    all_background_df = generate_newenv_profile(num, gen_model, temperature)
    columns = [ "pk",
                "codename",
                "scenario",
                "agent_goals",
                "relationship",
                "age_constraint",
                "occupation_constraint",
                "source"]
    background_df = all_background_df[columns]
    envs = cast(list[dict[str, Any]], background_df.to_dict(orient="records"))
    filtered_envs = []
    filtered_envs_pks = []
    for env in envs:
        # in case the env["agent_goals"] is string, convert into list
        if isinstance(env["agent_goals"], str):
            env["agent_goals"] = ast.literal_eval(env["agent_goals"])
        assert isinstance(env["relationship"], int)
        if len(env["agent_goals"]) == 2:
            env_pk = env["pk"]
            env.pop("pk")
            filtered_envs.append(env)
            filtered_envs_pks.append(env_pk)
    # add to database
    env_profiles = add_env_profiles(filtered_envs)

    # print(env_profiles)
    # also save new combo to database
    for env_profile in env_profiles:
        assert env_profile.pk is not None
        _sample_env_agent_combo_and_push_to_db(env_profile.pk)

    Migrator().run()

if __name__ == "__main__":
    app()