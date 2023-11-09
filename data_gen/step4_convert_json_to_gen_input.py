import json
import jsonlines

format_instruction = 'Your available action types are\n\"none action speak non-verbal communication leave\".\nNote: You can \"leave\" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.\n\nPlease only generate a JSON string including the action type and the argument.\nYour action should follow the given format:\n\nAs an example, for the schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}\nthe object {\"foo\": [\"bar\", \"baz\"]} is a well-formatted instance of the schema. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is not well-formatted.\n\nHere is the output schema:\n\\n{\\"description\\": \\"An interface for messages.\\\\nThere is only one required method: to_natural_language\\", \\"properties\\": {\\"action_type\\": {\\"title\\": \\"Action Type\\", \\"description\\": \\"whether to speak at this turn or choose to not do anything\\", \\"enum\\": [\\"none\\", \\"speak\\", \\"non-verbal communication\\", \\"action\\", \\"leave\\"], \\"type\\": \\"string\\"}, \\"argument\\": {\\"title\\": \\"Argument\\", \\"description\\": \\"the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action\\", \\"type\\": \\"string\\"}}, \\"required\\": [\\"action_type\\", \\"argument\\"]}\\n\u001b[0m\n'


def get_agent_info(agent1_pk, agent2_pk, env_pk, agent_dict, env_dict):
    agent1_name = agent_dict[agent1_pk]['first_name'] + ' ' + agent_dict[agent1_pk]['last_name']
    agent2_name = agent_dict[agent2_pk]['first_name'] + ' ' + agent_dict[agent2_pk]['last_name']

    agent1_age = agent_dict[agent1_pk]['age']
    agent2_age = agent_dict[agent2_pk]['age']

    agent1_occupation = agent_dict[agent1_pk]['occupation']
    agent2_occupation = agent_dict[agent2_pk]['occupation']

    if agent_dict[agent1_pk]['gender'] == 'Man':
        agent1_gender = 'male'
    elif agent_dict[agent1_pk]['gender'] == 'Woman':
        agent1_gender = 'female'
    elif agent_dict[agent1_pk]['gender'] == 'Nonbinary':
        agent1_gender = 'nonbinary'
    
    # agent2 the same
    if agent_dict[agent2_pk]['gender'] == 'Man':
        agent2_gender = 'male'
    elif agent_dict[agent2_pk]['gender'] == 'Woman':
        agent2_gender = 'female'
    elif agent_dict[agent2_pk]['gender'] == 'Nonbinary':
        agent2_gender = 'nonbinary'    

    agent1_public_info = agent_dict[agent1_pk]['public_info']
    agent2_public_info = agent_dict[agent2_pk]['public_info']

    agent1_personality_and_values = agent_dict[agent1_pk]['personality_and_values']
    agent2_personality_and_values = agent_dict[agent2_pk]['personality_and_values']

    agent1_secret = agent_dict[agent1_pk]['secret']
    agent2_secret = agent_dict[agent2_pk]['secret']

    agent1_goal = env_dict[env_pk]['agent_goals'][0].replace('<extra_info>', '')
    agent2_goal = env_dict[env_pk]['agent_goals'][1].replace('<extra_info>', '')

    agent1_info = {
        'agent_name': agent1_name,
        'agent_age': agent1_age,
        'agent_occupation': agent1_occupation,
        'agent_gender': agent1_gender,
        'agent_public_info': agent1_public_info,
        'agent_personality_and_values': agent1_personality_and_values,
        'agent_secret': agent1_secret,
        'agent_goal': agent1_goal,
    }

    agent2_info = {
        'agent_name': agent2_name,
        'agent_age': agent2_age,
        'agent_occupation': agent2_occupation,
        'agent_gender': agent2_gender,
        'agent_public_info': agent2_public_info,
        'agent_personality_and_values': agent2_personality_and_values,
        'agent_secret': agent2_secret,
        'agent_goal': agent2_goal,
    }
    return agent1_info, agent2_info


def fill_template(agent1_info, agent2_info, scenario):
    # Assuming the scenario is a string that is passed to the function
    # Gender pronouns are typically 'he/him', 'she/her', 'they/them', etc.
    # I'm adding placeholders for these pronouns; you'll need to replace them with actual values.
    agent1_pronoun = "their" # Replace with actual pronoun
    agent2_pronoun = "their" # Replace with actual pronoun

    prompt_template = (
        "Prompt after formatting:\n"
        "Imagine you are {agent1_name}, your task is to act/speak as {agent1_name} would, "
        "keeping in mind {agent1_name}s social goal.\n"
        "You can find {agent1_name}'s background and goal in the 'Here is the context of the interaction' field.\n"
        "Note that {agent1_name}'s secret and goal is only visible to you.\n"
        "You should try your best to achieve {agent1_name}'s goal in a way that align with their character traits.\n"
        "Additionally, maintaining the conversation's naturalness and realism is essential "
        "(e.g., do not repeat what other people has already said before).\n\n"
        "Here is the context of this interaction:\n"
        "Scenario: {scenario}\n"
        "{agent1_name}'s background: {agent1_name} is a {agent1_age}-year-old {agent1_gender} {agent1_occupation}. "
        "{agent1_pronoun} pronouns. {agent1_public_info} "
        "Personality and values description: {agent1_personality_and_values} "
        "{agent1_name}'s secrets: {agent1_secret}\n"
        "{agent2_name}'s goal: Unknown\n"
        "{agent1_name}'s goal: {agent1_goal}\n"
        "Conversation Starts:\n.\nYou are at Turn #0."
    )

    prompt = prompt_template.format(
        agent1_name=agent1_info['agent_name'],
        agent1_age=agent1_info['agent_age'],
        agent1_gender=agent1_info['agent_gender'],
        agent1_occupation=agent1_info['agent_occupation'],
        agent1_pronoun=agent1_pronoun,
        agent1_public_info=agent1_info['agent_public_info'],
        agent1_personality_and_values=agent1_info['agent_personality_and_values'],
        agent1_secret=agent1_info['agent_secret'],
        agent1_goal=agent1_info['agent_goal'],
        agent2_name=agent2_info['agent_name'],
        agent2_age=agent2_info['agent_age'],
        agent2_gender=agent2_info['agent_gender'],
        agent2_occupation=agent2_info['agent_occupation'],
        agent2_pronoun=agent2_pronoun,
        agent2_public_info=agent2_info['agent_public_info'],
        agent2_personality_and_values=agent2_info['agent_personality_and_values'],
        scenario=scenario
    )

    return prompt + format_instruction




with open('redis_json_data.json', 'r') as f:
    all_json_data = json.load(f)

agent_dict = {}
env_dict = {}
for key, data in all_json_data.items():
    if 'AgentProfile' in key:
        agent_dict[data['pk']] = data
    if 'EnvironmentProfile' in key:
        env_dict[data['pk']] = data

full_prompts = []
for key, data in all_json_data.items():
    #if data['pk'] != "01HER590MH0W1TPCPYKCAWMNXW":
    #    continue
    if 'EnvAgentComboStorage' in key:
        env_id = data['env_id']
        agent_ids = data['agent_ids']
        agent1_info, agent2_info = get_agent_info(agent_ids[0], agent_ids[1], env_id, agent_dict, env_dict)
        full_prompt = fill_template(agent1_info, agent2_info, env_dict[env_id]['scenario'])
        full_prompts.append({'text': full_prompt})
        full_prompt = fill_template(agent2_info, agent1_info, env_dict[env_id]['scenario'])
        full_prompts.append({'text': full_prompt})

print('Total number of prompts: ', len(full_prompts))
with jsonlines.open('full_prompts.jsonl', 'w') as writer:
    writer.write_all(full_prompts)
