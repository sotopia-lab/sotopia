from sotopia.database.persistent_profile import EnvironmentList
import json, os
from sotopia.database import EnvironmentProfile, AgentProfile
from sotopia.database.persistent_profile import RelationshipType

def create_agents(agent_profiles: list[dict]):
    final_profiles = []
    for agent in agent_profiles:
        found_profiles = None
        # compose AgentProfile[key] == value for all keys in agent
        all_finds = []
        for k, v in agent.items():
            if isinstance(v, str):
                v = v.replace("\n", "")
            if k in ["first_name", "last_name", "age", "personality_and_values"]:
                found_profiles = eval(f'AgentProfile.find(AgentProfile.{k} == "{v}").all()')
                all_finds.append([profile.pk for profile in found_profiles])
        
        # get the intersection of all finds
        final_profile = list(set.intersection(*map(set, all_finds)))
        if len(final_profile) == 0:
            print(f"Creating new AgentProfile: {agent}")
            final_profile = AgentProfile(**agent)
        else:
            if len(final_profile) > 1:
                print("Multiple profiles found", final_profile)
            # assert len(final_profile) == 1, f"Multiple profiles found: {final_profile}"
            profile = AgentProfile.get(final_profile[0])
            final_profile = profile
            print("Found profile", profile)
            # for k, v in agent.items():
            #     profile.__setattr__(k, v)
            # profile.save()
            # final_profile = profile.pk
            # print("Updated profile", profile)
            
        final_profiles.append(final_profile)
    return final_profiles

def create_environments(environment_profiles: list[dict], list_name: str):
    final_profiles = []
    for env in environment_profiles:
        found_profiles = None
        # compose AgentProfile[key] == value for all keys in agent
        all_finds = []
        # for k, v in env.items():
        #     if isinstance(v, str):
        #         v = v.replace("\n", "")
        #     found_profiles = EnvironmentProfile.find(getattr(EnvironmentProfile, k) == v).all()
        #     # found_profiles = eval(f'EnvironmentProfile.find(EnvironmentProfile.{k} == {repr(v)}).all()')
        #     all_finds.append([episode.pk for episode in found_profiles])
        
        # get the intersection of all finds
        final_profile = list(set.intersection(*map(set, all_finds))) if all_finds != [] else []
        if len(final_profile) == 0:
            print(f"Creating new EnvironmentProfile: {env}")
            env["codename"] = f"{list_name}_" + env["codename"]
            final_profile = EnvironmentProfile(**{**env, "relationship": RelationshipType.know_by_name, "age_constraint": "[(18, 70), (18, 70)]", "occupation_constraint": "[['Hiring Manager'], ['Candidate']]"})
        else:
            assert len(final_profile) == 1, f"Multiple profiles found: {final_profile}"
            profile = EnvironmentProfile.get(final_profile[0])
            final_profile = profile
            
        final_profiles.append(final_profile)
    return final_profiles
        

def generate_env_agent_list_hiring_exercise(envs: list[dict], agents: list[dict], list_name: str) -> EnvironmentList:
    agent_profiles = create_agents(agents)
    environment_profiles = create_environments(envs, list_name)
    
    manager_profiles = [agent for agent in agent_profiles if agent.occupation == "Hiring Manager"]
    candidate_profiles = [agent for agent in agent_profiles if agent.occupation == "Candidate"]
    assert len(manager_profiles) > 0, "No manager profiles found"
    assert len(candidate_profiles) > 0, "No candidate profiles found"
    
    profile_combinations = [
        (manager, candidate) for manager in manager_profiles for candidate in candidate_profiles
    ]
    env_profile_combinations: list[tuple[EnvironmentProfile, tuple[AgentProfile, AgentProfile]]] = [
        (env, profile_comb) for env in environment_profiles for profile_comb in profile_combinations
    ]
    
    environment_lists = EnvironmentList(
        name=list_name,
        environments=[env.pk for env, profile_comb in env_profile_combinations],
        agent_index=[f"{profile_comb[0].pk}_{profile_comb[1].pk}" for env, profile_comb in env_profile_combinations]
    )
    
    assert EnvironmentList.find(EnvironmentList.name == list_name).all() == [], f"EnvironmentList {list_name} already exists"
    for env, profile_comb in env_profile_combinations:
        env.save()
        for profile in profile_comb:
            profile.save()
    environment_lists.save()
    print(f"EnvironmentList {environment_lists.pk} with name {environment_lists.name} saved, with total environments {len(env_profile_combinations)}")
    return environment_lists

def generate_env_agent_list_cybersec(envs: list[dict], agents: list[dict], list_name: str) -> None:
    agent_profiles = create_agents(agents)
    environment_profiles = create_environments(envs, list_name)
    for profile in environment_profiles:
        profile.save()
    
    # target_condition = [AgentProfile.first_name == "Noah", AgentProfile.last_name == "Davis"]
    # all_managers = []
    # for condition in target_condition:
    #     all_pks = [profile.pk for profile in AgentProfile.find(condition).all()]
    #     all_managers.append(all_pks)
    # manager_profiles = [AgentProfile.get(list(set.intersection(*map(set, all_managers))))]
    
    
    # target_condition = [AgentProfile.first_name == "Jasmine", AgentProfile.last_name == "Blake"]
    # all_candidates = []
    # for condition in target_condition:
    #     all_pks = [profile.pk for profile in AgentProfile.find(condition).all()]
    #     all_candidates.append(all_pks)
    
    # candidate_profiles = [AgentProfile.get(list(set.intersection(*map(set, all_candidates))))]
    # assert len(manager_profiles) > 0, "No manager profiles found"
    # assert len(candidate_profiles) > 0, "No candidate profiles found"
    
    # profile_combinations = [
    #     (manager, candidate) for manager in manager_profiles for candidate in candidate_profiles
    # ]
    # env_profile_combinations: list[tuple[EnvironmentProfile, tuple[AgentProfile, AgentProfile]]] = [
    #     (env, profile_comb) for env in environment_profiles for profile_comb in profile_combinations
    # ]
    
    # environment_lists = EnvironmentList(
    #     name=list_name,
    #     environments=[env.pk for env, profile_comb in env_profile_combinations],
    #     agent_index=[f"{profile_comb[0].pk}_{profile_comb[1].pk}" for env, profile_comb in env_profile_combinations]
    # )
    
    # assert EnvironmentList.find(EnvironmentList.name == list_name).all() == [], f"EnvironmentList {list_name} already exists"
    # for env, profile_comb in env_profile_combinations:
    #     env.save()
    #     for profile in profile_comb:
    #         profile.save()
    # environment_lists.save()
    # print(f"EnvironmentList {environment_lists.pk} with name {environment_lists.name} saved")
    # return environment_lists


def main() -> None:
    # Usage: python sample_and_upload_to_env.py --name 1019_hiring_equal_competitive_salary_start_date --environment_file job_scenarios_bot_0922_salary_start_date_equal_competitive.json  --agent_file human_agreeableness_ai_all.json
    # Usage: python sample_and_upload_to_env.py --name 1019_hiring_equal_cooperative_salary_start_date --environment_file job_scenarios_bot_0922_salary_start_date_equal_cooperative.json  --agent_file human_agreeableness_ai_all.json
    import argparse
    parser = argparse.ArgumentParser(description="Sample and upload to env")
    parser.add_argument("--name", type=str, help="name of the environment")
    parser.add_argument("--environment_file", type=str, help="list of environments")
    parser.add_argument("--agent_file", type=str, help="list of agents")
    
    args = parser.parse_args()
    envs = json.load(open(args.environment_file))
    agents = json.load(open(args.agent_file))
    
    if "hiring" in args.name:
        generate_env_agent_list_hiring_exercise(envs, agents, args.name)
    elif "cybersec" in args.name:
        generate_env_agent_list_cybersec(envs, agents, args.name)

if __name__ == "__main__":
    main()