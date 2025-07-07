from sotopia.database.persistent_profile import EnvironmentList
import json
from sotopia.database import EnvironmentProfile, AgentProfile
from sotopia.database.persistent_profile import RelationshipType
import uuid

def create_agents(agent_profiles: list[dict]):
    """
    Create or update AgentProfile entries based on provided dicts.
    Supports explicit 'agent_id' or matching on core fields.
    """
    final_profiles = []
    print(f"Starting to create/update {len(agent_profiles)} agent(s)")
    for agent in agent_profiles:
        agent_id = agent.get("agent_id")
        if agent_id:
            try:
                profile = AgentProfile.get(agent_id)
                print(f"Found existing AgentProfile by id {agent_id}")
                for k, v in agent.items():
                    if k == "agent_id":
                        continue
                    if hasattr(profile, k):
                        setattr(profile, k, v)
                profile.save()
                print(f"Updated AgentProfile pk={profile.pk}")
            except Exception:
                print(f"Creating new AgentProfile with id {agent_id}")
                init_data = {k: v for k, v in agent.items() if k != "agent_id"}
                profile = AgentProfile(pk=agent_id, **init_data)
                profile.save()
                print(f"Created AgentProfile pk={profile.pk}")
            final_profiles.append(profile)
            continue

        # match by core fields
        search_fields = ["first_name", "last_name", "age", "occupation"]
        find_sets = []
        for k in search_fields:
            if k in agent:
                v = agent[k]
                if isinstance(v, str):
                    v = v.replace("\n", "")
                matches = AgentProfile.find(getattr(AgentProfile, k) == v).all()
                find_sets.append({p.pk for p in matches})
        intersect = set.intersection(*find_sets) if find_sets else set()
        if not intersect:
            print(f"Creating new AgentProfile: {agent}")
            profile = AgentProfile(**agent)
            profile.save()
            print(f"Created AgentProfile pk={profile.pk}")
        else:
            pk = intersect.pop()
            profile = AgentProfile.get(pk)
            print(f"Found profile pk={pk}")
            for k, v in agent.items():
                if hasattr(profile, k):
                    setattr(profile, k, v)
            profile.save()
            print(f"Updated AgentProfile pk={profile.pk}")
        final_profiles.append(profile)
    return final_profiles

def create_environments(environment_profiles: list[dict], list_name: str):
    """
    Create or update EnvironmentProfile entries based on provided dicts.
    Supports explicit 'env_id' or matching on 'codename'.
    Filters out any non-list 'agent_constraint' to avoid validation errors.
    """
    final_profiles = []
    print(f"Starting to create/update {len(environment_profiles)} environment(s) in list '{list_name}'")
    for env in environment_profiles:
        rel_val = env.get("relationship")
        rel_enum = RelationshipType[rel_val] if isinstance(rel_val, str) and rel_val in RelationshipType.__members__ else RelationshipType.know_by_name
        age_constr = env.get("age_constraint", "[(18, 70), (18, 70)]")
        occ_constr = env.get("occupation_constraint", "[['Hiring Manager'], ['Candidate']]")
        raw_agent_constr = env.get("agent_constraint")
        agent_constr = raw_agent_constr if isinstance(raw_agent_constr, list) else None
        codename = env.get("codename")
        env_id = env.get("env_id")

        # base_data excludes env_id and agent_constraint
        base_data = {k: v for k, v in env.items() if k not in ("env_id", "agent_constraint")}

        # by ID
        if env_id:
            try:
                profile = EnvironmentProfile.get(env_id)
                print(f"Found existing EnvironmentProfile by id {env_id}")
                for k, v in base_data.items():
                    if hasattr(profile, k):
                        setattr(profile, k, v)
                profile.relationship = rel_enum
                profile.age_constraint = age_constr
                profile.occupation_constraint = occ_constr
                if agent_constr is not None:
                    profile.agent_constraint = agent_constr
                profile.save()
                print(f"Updated EnvironmentProfile pk={profile.pk}")
                final_profiles.append(profile)
                continue
            except Exception:
                print(f"Creating new EnvironmentProfile with id {env_id}")
                init_data = {
                    **base_data,
                    "pk": env_id,
                    "relationship": rel_enum,
                    "age_constraint": age_constr,
                    "occupation_constraint": occ_constr
                }
                if agent_constr is not None:
                    init_data["agent_constraint"] = agent_constr
                init_data["codename"] = f"{list_name}_{codename or ''}"
                profile = EnvironmentProfile(**init_data)
                profile.save()
                print(f"Created EnvironmentProfile pk={profile.pk}")
                final_profiles.append(profile)
                continue

        # by codename
        matches = EnvironmentProfile.find(EnvironmentProfile.codename == codename).all() if codename else []
        if matches:
            profile = matches[0]
            print(f"Found profile by codename '{codename}' (pk={profile.pk})")
            for k, v in base_data.items():
                if hasattr(profile, k):
                    setattr(profile, k, v)
            profile.relationship = rel_enum
            profile.age_constraint = age_constr
            profile.occupation_constraint = occ_constr
            if agent_constr is not None:
                profile.agent_constraint = agent_constr
            profile.save()
            print(f"Updated EnvironmentProfile pk={profile.pk}")
        else:
            print(f"Creating new EnvironmentProfile: {env}")
            init_data = {
                **base_data,
                "relationship": rel_enum,
                "age_constraint": age_constr,
                "occupation_constraint": occ_constr
            }
            if agent_constr is not None:
                init_data["agent_constraint"] = agent_constr
            init_data["codename"] = f"{list_name}_{codename or ''}"
            profile = EnvironmentProfile(**init_data)
            profile.save()
            print(f"Created EnvironmentProfile pk={profile.pk}")
        final_profiles.append(profile)
    return final_profiles

def generate_env_agent_list_hiring_exercise(envs: list[dict], agents: list[dict], list_name: str) -> EnvironmentList:
    """Pair Hiring Manager & Candidate agents for each environment"""
    print(f"Generating hiring list '{list_name}'")
    agent_profiles = create_agents(agents)
    environment_profiles = create_environments(envs, list_name)
    managers = [a for a in agent_profiles if a.occupation == "Hiring Manager"]
    candidates = [a for a in agent_profiles if a.occupation == "Candidate"]
    assert managers and candidates, "Need at least one manager and one candidate"
    pairs = [(env, m, c) for env in environment_profiles for m in managers for c in candidates]

    print("PK:", list_name)
    print("Environments:", [e.pk for e, _, _ in pairs])
    print("Agent index:", [f"{m.pk}_{c.pk}" for _, m, c in pairs])
    print(EnvironmentList)

    env_list = EnvironmentList(
        pk=str(uuid.uuid4()),
        name=list_name,
        environments=[e.pk for e, _, _ in pairs],
        agent_index=[f"{m.pk}_{c.pk}" for _, m, c in pairs]
    )
    assert not EnvironmentList.find(EnvironmentList.name == list_name).all(), \
        f"EnvironmentList {list_name} already exists"
    print(f"Saving {len(pairs)} pairs to '{list_name}'")
    for idx, (e, m, c) in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] Env pk={e.pk}, M={m.pk}, C={c.pk}")
        e.save()
        m.save()
        c.save()
    env_list.save()
    print(f"List saved: pk={env_list.pk}")
    return env_list

def generate_env_agent_list_liedar_experiment(envs: list[dict], agents: list[dict], list_name: str) -> EnvironmentList:
    """Custom uploader for test_transparency_liedar_experiment_new"""
    print(f"Generating LIEdar experiment list '{list_name}'")
    agent_profiles = create_agents(agents)
    print("Length of Agent Profiles:", len(agent_profiles))
    environment_profiles = create_environments(envs, list_name)
    print("Length of Environment Profiles:", len(environment_profiles))

    manager_profiles = agent_profiles
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
    candidate_profiles = [a for a in agent_profiles if a.pk in allowed_pks]
    print("Managers:", len(manager_profiles), "Candidates:", len(candidate_profiles))
    assert manager_profiles, "No manager profiles found"
    assert candidate_profiles, "No candidate profiles found"

    profile_combinations = [(m, c) for m in manager_profiles for c in candidate_profiles]
    env_profile_combinations = [
        (env, combo) for env in environment_profiles for combo in profile_combinations
    ]
    print("Environment-Profile Combinations:", len(env_profile_combinations))
    print("Env pks:", [env.pk for env, _ in env_profile_combinations])
    print("Agent Index:", [f"{m.pk}_{c.pk}" for env, (m, c) in env_profile_combinations])
    print("pk", list_name)
    environment_lists = EnvironmentList(
        pk=str(uuid.uuid4()),
        name=list_name,
        environments=[env.pk for env, _ in env_profile_combinations],
        agent_index=[f"{m.pk}_{c.pk}" for env, (m, c) in env_profile_combinations]
    )
    assert not EnvironmentList.find(EnvironmentList.name == list_name).all(), \
        f"EnvironmentList {list_name} already exists"

    environment_lists.save()
    print(f"EnvironmentList {environment_lists.pk} with name {environment_lists.name} saved, "
          f"with total environments {len(env_profile_combinations)}")
    return environment_lists

def generate_env_agent_list_generic(envs: list[dict], agents: list[dict], list_name: str) -> EnvironmentList:
    """Pair every agent with every environment (fallback)"""
    print(f"Generating generic list '{list_name}'")
    agent_profiles = create_agents(agents)
    environment_profiles = create_environments(envs, list_name)
    pairs = [(env, a) for env in environment_profiles for a in agent_profiles]
    env_list = EnvironmentList(
        name=list_name,
        environments=[e.pk for e, a in pairs],
        agent_index=[str(a.pk) for e, a in pairs]
    )
    assert not EnvironmentList.find(EnvironmentList.name == list_name).all(), \
        f"EnvironmentList {list_name} already exists"
    print(f"Saving {len(pairs)} env-agent entries to '{list_name}'")
    for idx, (e, a) in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] Env pk={e.pk}, Agent pk={a.pk}")
        # No need to save here; already saved in create_agents/create_environments
    env_list.save()
    print(f"List saved: pk={env_list.pk}")
    return env_list

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sample and upload to env")
    parser.add_argument("--name", required=True)
    parser.add_argument("--environment_file", help="Path to environment JSON file")
    parser.add_argument("--agent_file", required=True)
    args = parser.parse_args()

    agents = json.load(open(args.agent_file))
    envs = json.load(open(args.environment_file)) if args.environment_file else []

    if args.name == "test_transparency_liedar_exp1":
        generate_env_agent_list_liedar_experiment(envs, agents, args.name)
    elif "hiring" in args.name:
        generate_env_agent_list_hiring_exercise(envs, agents, args.name)
    elif "cybersec" in args.name:
        # implement cybersec logic as needed
        pass
    else:
        generate_env_agent_list_generic(envs, agents, args.name)

if __name__ == "__main__":
    main()