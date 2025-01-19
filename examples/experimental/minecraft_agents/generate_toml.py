import pandas as pd

xlsx_file = 'examples\\experimental\\group_discussion_agents\\toml_generation.xlsx'
df = pd.read_excel(xlsx_file)

toml_template = '''redis_url = "redis://localhost:6379/0"
extra_modules = ["examples.experimental.group_discussion_agents.group_discussion_agents"]

[[nodes]]
node_name = "Jack"
node_class = "llm_agent"

[nodes.node_args]
query_interval = 15
output_channel = "Jack"
input_text_channels = ["Jane", "John"]
input_tick_channel = "tick/secs/1"
goal = """{goal_jack}"""
model_name = "gpt-4o-2024-11-20"
agent_name = "Jack"

[[nodes]]
node_name = "Jane"
node_class = "llm_agent"

[nodes.node_args]
query_interval = 19
output_channel = "Jane"
input_text_channels = ["Jack", "John"]
input_tick_channel = "tick/secs/1"
goal = """{goal_jane}"""
model_name = "gpt-4o-2024-11-20"
agent_name = "Jane"

[[nodes]]
node_name = "John"
node_class = "llm_agent"

[nodes.node_args]
query_interval = 23
output_channel = "John"
input_text_channels = ["Jack", "Jane"]
input_tick_channel = "tick/secs/1"
goal = """{goal_john}"""
model_name = "gpt-4o-2024-11-20"
agent_name = "John"

[[nodes]]
node_name = "record"
node_class = "record"

[nodes.node_args]
jsonl_file_path = "log.jsonl"

[nodes.node_args.record_channel_types]
"Jack" = "agent_action"
"Jane" = "agent_action"
"John" = "agent_action"

[[nodes]]
node_name = "tick"
node_class = "tick"
'''

for index, row in df.iterrows():
    goal_jack = row['Jack']
    goal_jane = row['Jane']
    goal_john = row['John']
    filename = row['Filename']
    
    toml_content = toml_template.format(
        goal_jack=goal_jack,
        goal_jane=goal_jane,
        goal_john=goal_john
    )
    
    output_filename = f"group_discussion_agents_{filename}.toml"
    with open(output_filename, 'w') as f:
        f.write(toml_content)