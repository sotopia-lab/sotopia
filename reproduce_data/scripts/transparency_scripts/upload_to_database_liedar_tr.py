from sotopia.database.persistent_profile import EnvironmentList
import json, os
from sotopia.database import EnvironmentProfile, AgentProfile
from sotopia.database.persistent_profile import RelationshipType

def create_agents(agent_profiles: list[dict]):
    """
    FIXED: Properly handle commas and quotes in personality_and_values
    """
    final_profiles = []
    for agent in agent_profiles:
        found_profiles = None 
        all_finds = []
        for k, v in agent.items():
            if k in ["first_name", "last_name", "age", "personality_and_values"]:
                if isinstance(v, str):
                    # FIXED: Properly escape both quotes AND handle commas
                    v_cleaned = v.replace("\n", "")
                    # Escape both single and double quotes for eval safety
                    v_escaped = v_cleaned.replace("'", "\\'").replace('"', '\\"')
                    
                    # Use repr() instead of manual quoting to handle complex strings
                    found_profiles = eval(f'AgentProfile.find(AgentProfile.{k} == {repr(v_cleaned)}).all()')
                elif isinstance(v, int):
                    # Handle integers like age
                    found_profiles = eval(f'AgentProfile.find(AgentProfile.{k} == {v}).all()')
                else:
                    # Handle other types
                    found_profiles = eval(f'AgentProfile.find(AgentProfile.{k} == {repr(v)}).all()')
                
                all_finds.append([profile.pk for profile in found_profiles])
        
        # Get intersection of all finds
        final_profile = list(set.intersection(*map(set, all_finds))) if all_finds else []
        
        if len(final_profile) == 0:
            print(f"Creating new AgentProfile: {agent}")
            
            # Handle agent_id -> pk conversion for existing human agents
            agent_clean = agent.copy()
            if 'agent_id' in agent_clean:
                agent_clean['pk'] = agent_clean.pop('agent_id')
            
            # Log transparency for debugging
            personality = agent.get('personality_and_values', '')
            if 'High Transparency' in personality:
                print(f"  -> Creating HIGH transparency agent")
            elif 'Low Transparency' in personality:
                print(f"  -> Creating LOW transparency agent")
            
            final_profile = AgentProfile(**agent_clean)
        else:
            if len(final_profile) > 1:
                print("Multiple profiles found", final_profile)
            profile = AgentProfile.get(final_profile[0])
            final_profile = profile
            print("Found profile", profile)
            
        final_profiles.append(final_profile)
    
    return final_profiles

def create_environments(environment_profiles: list[dict], list_name: str):
    """
    Keep the same as reference file - it works
    """
    final_profiles = []
    for env in environment_profiles:
        found_profiles = None
        all_finds = []
        
        # Get intersection of all finds
        final_profile = list(set.intersection(*map(set, all_finds))) if all_finds != [] else []
        if len(final_profile) == 0:
            print(f"Creating new EnvironmentProfile: {env}")
            env["codename"] = f"{list_name}_" + env["codename"]
            # MODIFIED: Use the constraints from your debug script
            final_profile = EnvironmentProfile(**{
                **env, 
                "relationship": RelationshipType.know_by_name, 
                "age_constraint": "[(18, 70), (18, 70)]", 
                "occupation_constraint": None,  # Changed from reference
                "agent_constraint": None        # Added from your version
            })
        else:
            assert len(final_profile) == 1, f"Multiple profiles found: {final_profile}"
            profile = EnvironmentProfile.get(final_profile[0])
            final_profile = profile
            
        final_profiles.append(final_profile)
    return final_profiles

def generate_env_agent_list_hiring_exercise(envs: list[dict], agents: list[dict], list_name: str) -> EnvironmentList:
    """
    Modified from reference to handle your specific agent structure
    """
    agent_profiles = create_agents(agents)
    environment_profiles = create_environments(envs, list_name)
    
    # MODIFIED: Handle your specific agent setup
    manager_profiles = [agent for agent in agent_profiles if agent.occupation == "Hiring Manager"]
    
    # Your allowed candidate PKs
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
    
    print(f"Managers: {len(manager_profiles)}, Candidates: {len(candidate_profiles)}")
    
    assert len(manager_profiles) > 0, "No manager profiles found"
    assert len(candidate_profiles) > 0, "No candidate profiles found"
    
    # Rest is same as reference
    profile_combinations = [
        (manager, candidate) for manager in manager_profiles for candidate in candidate_profiles
    ]
    env_profile_combinations = [
        (env, profile_comb) for env in environment_profiles for profile_comb in profile_combinations
    ]
    
    environment_lists = EnvironmentList(
        name=list_name,
        environments=[env.pk for env, profile_comb in env_profile_combinations],
        agent_index=[f"{profile_comb[0].pk}_{profile_comb[1].pk}" for env, profile_comb in env_profile_combinations]
    )
    
    # Check if exists (but don't fail hard)
    existing = EnvironmentList.find(EnvironmentList.name == list_name).all()
    if existing:
        print(f"WARNING: EnvironmentList {list_name} already exists")
    
    # Save everything
    for env, profile_comb in env_profile_combinations:
        env.save()
        for profile in profile_comb:
            profile.save()
    environment_lists.save()
    
    print(f"EnvironmentList {environment_lists.pk} with name {environment_lists.name} saved")
    print(f"Total environments: {len(env_profile_combinations)}")
    return environment_lists

# Keep the rest same as reference
def generate_env_agent_list_cybersec(envs: list[dict], agents: list[dict], list_name: str) -> None:
    agent_profiles = create_agents(agents)
    environment_profiles = create_environments(envs, list_name)
    for profile in environment_profiles:
        profile.save()

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Sample and upload to env")
    parser.add_argument("--name", type=str, help="name of the environment")
    parser.add_argument("--environment_file", type=str, help="list of environments")
    parser.add_argument("--agent_file", type=str, help="list of agents")
    
    args = parser.parse_args()
    envs = json.load(open(args.environment_file))
    agents = json.load(open(args.agent_file))
    print("Loaded environment and agent files")
    print(f"Environments: {len(envs)}, Agents: {len(agents)}")
    
    if "hiring" in args.name or "liedar" in args.name or "transparency" in args.name:
        generate_env_agent_list_hiring_exercise(envs, agents, args.name)
    elif "cybersec" in args.name:
        generate_env_agent_list_cybersec(envs, agents, args.name)

if __name__ == "__main__":
    main()