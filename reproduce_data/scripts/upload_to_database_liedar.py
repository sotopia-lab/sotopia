"""
You initially load the environment and agent profiles from the json files.
You then create the agent profiles and environment profiles. This changes dependent on what scenario you are using. What kind of scenarios does AI-Liedar have? Need to check this ###

This leads us to generate_env_agent_list_hiring_exercise(envs, agents, args.name) since we are only considering hiring. 
What does this function do?
This function helps create the agents and environment profiles. It then creates a combination of manager and candidate profiles that ensures that each manager is paired with each candidate.
It then creates a combination of environment profiles with the manager-candidate profile combinations.
This creates an environment list with the name, environment profiles, and agent index.
It then checks if the EnvironmentList already exists. If it does, we assert that it doesn't exist.
It then saves the environment list and prints the number of environments saved.
This is the final step where the database has the environmentlists we need.
You can think of an environment list as a list of environments and agents for a specific scenario you want to study.
"""

from sotopia.database.persistent_profile import EnvironmentList
import json, os
from sotopia.database import EnvironmentProfile, AgentProfile
from sotopia.database.persistent_profile import RelationshipType

def create_agents(agent_profiles: list[dict]):
    """
    Import a list of agent profiles. This is basically all the names etc. of the agents. 
    There is a setup of found profiles, which means that we check if the agent profile already exists.
    We do this by checking is the key-value input pairs we pass are already in the database. If it doesn't exist, we don't append it to the all_finds list.
    """
    final_profiles = []
    for agent in agent_profiles:
        found_profiles = None 
        # compose AgentProfile[key] == value for all keys in agent
        all_finds = []
        for k, v in agent.items():
            #What does this do? : If the value is a string, we replace the new line character with an empty string.
            if isinstance(v, str):
                v = v.replace("\n", "")
            if k in ["first_name", "last_name", "age", "personality_and_values"]:
                #What does eval do here? : It evaluates the string expression and returns the result. 
                #In this case, it returns all the agent profiles that have the key k equal to value v. If not found, it returns an empty list.
                #This is a way to search for the agent profiles that have the same key value pair as the agent profile we are looking for.
                found_profiles = eval(f'AgentProfile.find(AgentProfile.{k} == "{v}").all()')
                all_finds.append([profile.pk for profile in found_profiles])
        
        # What is the purpose of this? : We are looking for the intersection of all the agent profiles that have the same key-value pairs as the agent profile we are looking for.
        # This means that if the agent profile has parameters such as first_name, last_name, age, and personality_and_values, we are looking for the agent profile that has all these parameters.
        # If we find the agent profile, we append it to the final_profile list. If not, we create a new agent profile.
        final_profile = list(set.intersection(*map(set, all_finds)))
        if len(final_profile) == 0:
            print(f"Creating new AgentProfile: {agent}")
            final_profile = AgentProfile(**agent)
        else:
            if len(final_profile) > 1:
                print("Multiple profiles found", final_profile)
            # assert len(final_profile) == 1, f"Multiple profiles found: {final_profile}"
            #This takes the first agent profile that we found.
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
    """
    Here we are creating the environment profiles.
    We are usually creating a new profile. If we find a profile that matches the key-value pairs, we append it to the final_profiles list.
    When creating a new environment:
        - We have a code name for the environment 
        - We have a relationship type which can be changed and modified.
        - We have an age constraint which is a list of tuples. Each tuple is a range of age.
        - We have an occupation constraint which is a list of lists. Each list is a list of occupations.
    If found: we assert that the length of the final_profile is 1. If not, we print that multiple profiles are found.
    We then get the first one from the list of profiles found.
    """
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
            # final_profile = EnvironmentProfile(**{**env, "relationship": RelationshipType.know_by_name, "age_constraint": "[(18, 70), (18, 70)]", "occupation_constraint": "[['Hiring Manager'], ['Candidate']]"}).save()
            final_profile = EnvironmentProfile(**{**env, "relationship": RelationshipType.know_by_name, "age_constraint": "[(18, 70), (18, 70)]", "occupation_constraint": None, "agent_constraint": None}).save()
        else:
            assert len(final_profile) == 1, f"Multiple profiles found: {final_profile}"
            profile = EnvironmentProfile.get(final_profile[0])
            final_profile = profile
            
        final_profiles.append(final_profile)
    return final_profiles
        

def generate_env_agent_list_hiring_exercise(envs: list[dict], agents: list[dict], list_name: str) -> EnvironmentList:
    """
    This helps create an EnvironmentList for the hiring exercise, where an EnvironmentList is a list of environments and agent.
    This is where each environment is linked to an agent combo, and each agent gets paired with every other agent.
    """
    agent_profiles = create_agents(agents) 
    print("Length of Agent Profiles",len(agent_profiles)) ## This creates an agent profile if an exact match is not found in the database.
    environment_profiles = create_environments(envs, list_name) ## This creates an environment profile if an exact match is not found in the database.
    print("Length of Environment Profiles",len(environment_profiles)) ## This creates an environment profile if an exact match is not found in the database.
    manager_profiles = [agent for agent in agent_profiles if agent.occupation == "Hiring Manager"]
    allowed_pks = [
    '01H5TNE5PP870BS5HP2FPPKS2Y',
    '01H5TNE5PY896ASNX8XGQA6AE0',
    '01H5TNE5PWZ5PNDTGKDYRY36PQ',
    '01H5TNE5PT8KW11GZ99Q0T43V4',
    '01H5TNE5P90FYSTBMW5DG5ERCG',
    '01H5TNE5PJTHMQ1Q3T398YN990',
    '01H5TNE5PFT9HH0WRT6W1NY5GZ',
    '01H5TNE5PW9SZFM058Z8P7PR5C',
    '01H5TNE5P83CZ1TDBVN74NGEEJ',
    '01H5TNE5P7RVY0TYX8VTCXABR6',
    '01H5TNE5PDV7WZ0C5KTGGXX1NR',
    '01H5TNE5P8F9NJ2QK2YP5HPXKH',
    '01H5TNE5PN656EADK59K4DG793'
    ]

    candidate_profiles = [agent for agent in agent_profiles if agent.pk in allowed_pks]
    print(len(manager_profiles), len(candidate_profiles))
    
    assert len(manager_profiles) > 0, "No manager profiles found"
    assert len(candidate_profiles) > 0, "No candidate profiles found"
    # This creates a combination of manager and candidate profiles that ensures that each manager is paired with each candidate.
    profile_combinations = [
        (manager, candidate) for manager in manager_profiles for candidate in candidate_profiles
    ]
    # This creates a combination of environment profiles with the manager-candidate profile combinations.
    env_profile_combinations: list[tuple[EnvironmentProfile, tuple[AgentProfile, AgentProfile]]] = [
        (env, profile_comb) for env in environment_profiles for profile_comb in profile_combinations
    ]
    
    #This creates an environment list with the name, environment profiles, and agent index.
    environment_lists = EnvironmentList(
        name=list_name,
        environments=[env.pk for env, profile_comb in env_profile_combinations],
        agent_index=[f"{profile_comb[0].pk}_{profile_comb[1].pk}" for env, profile_comb in env_profile_combinations]
    )
    
    #Checking if the EnvironmentList already exists. If it does, we assert that it doesn't exist.
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
    print("Loaded")
    
    if "hiring" in args.name or "liedar" in args.name:
        generate_env_agent_list_hiring_exercise(envs, agents, args.name)
    elif "cybersec" in args.name:
        generate_env_agent_list_cybersec(envs, agents, args.name)

if __name__ == "__main__":
    main()