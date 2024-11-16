# OpenHands Interview Example

In this example, we explore the interaction between two LLM agents, "Jack" and "Jane", within an OpenHands runtime node. These agents are designed to simulate an interview scenario where "Jack" evaluates "Jane's" technical abilities and communication skills.

## Agents Overview

- **Jack (Interviewer)**: An LLM agent tasked with assessing Jane's technical skills, communication, problem-solving approach, and enthusiasm.
- **Jane (Interviewee)**: An LLM agent aiming to perform well in the interview by showcasing her skills and enthusiasm.

### Customizing the Scenario

You can modify the `interview_openhands.toml` file to simulate different scenarios by changing the goals, model names, and communication channels. This flexibility allows you to tailor the interaction to various use cases, such as customer service simulations, collaborative problem-solving, or educational tutoring.

### Actions Available to LLM Agents

- **speak**: Communicate with other agents by sending a message.
- **thought**: Make a plan or set a goal internally.
- **none**: Choose not to take any action, often used when waiting for data.
- **non-verbal**: Perform a non-verbal action, such as a gesture.
- **browse**: Open a web page.
- **browse_action**: Perform actions on a web browser, such as navigating or interacting with elements.
- **read**: Read the content of a file.
- **write**: Write content to a file.
- **run**: Execute a command in a Linux shell.
- **leave**: Exit the conversation when goals are completed or abandoned.

These actions enable the agents to interact dynamically with their environment and each other, providing a robust framework for simulating complex scenarios.

## Dataflow Configuration

The dataflow file `interview_openhands.toml` is used to configure the interaction between the agents. It acts as a blueprint, defining how the agents communicate and process information.

### Example Configuration

```toml
[[nodes]]
node_name = "Jack"
node_class = "llm_agent"

[nodes.node_args]
query_interval = 5
output_channel = "Jack:Jane"
input_text_channels = ["Jane:Jack"]
input_env_channels = ["Runtime:Agent"]
input_tick_channel = "tick/secs/1"
goal = "Your goal is to effectively test Jane's technical ability and finally decide if she has passed the interview. Make sure to also evaluate her communication skills, problem-solving approach, and enthusiasm."
model_name = "gpt-4o-mini"
agent_name = "Jack"

[[nodes]]
node_name = "Jane"
node_class = "llm_agent"

[nodes.node_args]
query_interval = 7
output_channel = "Jane:Jack"
input_text_channels = ["Jack:Jane"]
input_env_channels = ["Runtime:Agent"]
input_tick_channel = "tick/secs/1"
goal = "Your goal is to do well in the interview by demonstrating your technical skills, clear communication, and enthusiasm for the position. Stay calm, ask clarifying questions when needed, and confidently explain your thought process."
model_name = "gpt-4o-mini"
agent_name = "Jane"
```

### Auxiliary Nodes

In addition to the agents, there are auxiliary nodes that facilitate the interaction:

- **Tick Node**: Provides a stable clock signal for synchronization.
- **Print Node**: Outputs real-time messages for monitoring the interaction.

#### Auxiliary Node Configuration

```toml
[[nodes]]
node_name = "tick"
node_class = "tick"

[[nodes]]
node_name = "print"
node_class = "print"

[nodes.node_args.print_channel_types]
"tick/secs/1" = "tick"
"Jane:Jack" = "agent_action"
"Jack:Jane" = "agent_action"
```

## Running the Example

### Prerequisites

1. **Redis**: Ensure Redis is hosted at `localhost:6379`. Update the `redis_url` in the `interview_openhands.toml` file if using a different Redis database.

2. **Setting Up OpenHands Runtime**: Clone the repository and set up the environment.

   - **Clone the Repository**: Clone the repository from GitHub to get the necessary files and configurations.
     ```bash
     git clone https://github.com/akhatua2/aact-openhands
     cd aact-openhands
     ```

   - **Python and Poetry**: Ensure you have Python 3.12 installed. Install Poetry for dependency management if you haven't already.
     ```bash
     curl -sSL https://install.python-poetry.org | python3 -
     ```

   - **Install Dependencies**: Navigate to the project directory and install the dependencies using Poetry.
     ```bash
     poetry install
     ```

   - **Environment Variables**: Set up your environment variables. Copy the `.env.example` file to a new file named `.env` in the root of your project directory and replace the placeholder values with your actual API keys and URLs.
     ```bash
     cp .env.example .env
     ```

     Edit the `.env` file:
     ```plaintext
     MODAL_API_TOKEN_ID=your_actual_modal_api_token_id
     MODAL_API_TOKEN_SECRET=your_actual_modal_api_token_secret
     ```
   - **Running the OpenHands Node**

        To run the OpenHands node with the provided configuration, use the following command:

        ```bash
        poetry run aact run-dataflow examples/openhands_node.toml
        ```

        Upon successful execution, you should see output similar to the following:

        ```bash
        16:41:26 - openhands:INFO: openhands_node.py:120 - --------------------
        16:41:26 - openhands:INFO: openhands_node.py:121 - RUNTIME CONNECTED
        16:41:26 - openhands:INFO: openhands_node.py:122 - --------------------
        16:41:26 - openhands:INFO: openhands_node.py:127 - Runtime initialization took 157.77 seconds.
        ```

These steps ensure that you have the necessary environment and dependencies to run the OpenHands node and the example scenario.

### Command

To run this interview setting example, use the following command:

```bash
uv run aact run-dataflow examples/experimental/interview_openhands/interview_openhands.toml
```

### Expected Output

You will see JSON strings printed out, representing messages exchanged between the nodes. These messages include timestamps for easy debugging and data recording, demonstrating the real-time interaction between the agents.

```json
{"timestamp":"2024-11-12T17:27:27.797795","channel":"tick/secs/1","data":{"data_type":"tick","tick":0}}
{"timestamp":"2024-11-12T17:27:28.810269","channel":"tick/secs/1","data":{"data_type":"tick","tick":1}}
{"timestamp":"2024-11-12T17:27:29.805630","channel":"tick/secs/1","data":{"data_type":"tick","tick":2}}
{"timestamp":"2024-11-12T17:27:30.803932","channel":"tick/secs/1","data":{"data_type":"tick","tick":3}}
{"timestamp":"2024-11-12T17:27:31.804342","channel":"tick/secs/1","data":{"data_type":"tick","tick":4}}
{"timestamp":"2024-11-12T17:27:32.805544","channel":"tick/secs/1","data":{"data_type":"tick","tick":5}}
{"timestamp":"2024-11-12T17:27:33.094975","channel":"Jack:Jane","data":{"data_type":"agent_action","agent_name":"Jack","
action_type":"speak","argument":"Hi Jane! I'm Jack, a Principal Software Engineer here at Fintech. It's great to meet you! Could you please introduce yourself and tell me a little about your education and any relevant experiences you've had?","path":""}}
{"timestamp":"2024-11-12T17:27:33.801111","channel":"tick/secs/1","data":{"data_type":"tick","tick":6}}
{"timestamp":"2024-11-12T17:27:34.803471","channel":"tick/secs/1","data":{"data_type":"tick","tick":7}}
{"timestamp":"2024-11-12T17:27:35.118890","channel":"Jane:Jack","data":{"data_type":"agent_action","agent_name":"Jane","
action_type":"speak","argument":"Hi Jack! I'm Jane, a college senior at Stanford University majoring in Computer Science. I've had a strong focus on software development, especially in fintech applications. I recently completed a project where I built a budgeting app using React and Node.js, which helped users manage their finances effectively. I'm excited about the opportunity to bring my skills to your team.","path":""}}
{"timestamp":"2024-11-12T17:27:35.803424","channel":"tick/secs/1","data":{"data_type":"tick","tick":8}}
{"timestamp":"2024-11-12T17:27:36.800113","channel":"tick/secs/1","data":{"data_type":"tick","tick":9}}
{"timestamp":"2024-11-12T17:27:37.732772","channel":"Jack:Jane","data":{"data_type":"agent_action","agent_name":"Jack","
action_type":"speak","argument":"That's a fantastic introduction, Jane! Your project sounds really interesting. Now, let's dive into some technical questions. Are you ready?","path":""}}
{"timestamp":"2024-11-12T17:27:37.802020","channel":"tick/secs/1","data":{"data_type":"tick","tick":10}}
{"timestamp":"2024-11-12T17:27:38.800835","channel":"tick/secs/1","data":{"data_type":"tick","tick":11}}
{"timestamp":"2024-11-12T17:27:39.804363","channel":"tick/secs/1","data":{"data_type":"tick","tick":12}}
```

This output shows the following events:

1. The tick node sends a tick message, indicating the passage of time.
2. "Jack" initiates the conversation by introducing himself and asking "Jane" to introduce herself.
3. "Jane" responds with her introduction, highlighting her education and relevant experiences.
4. "Jack" acknowledges Jane's introduction and prepares to ask technical questions.

Each message includes a timestamp for easy debugging and data recording. The interaction demonstrates the agents' ability to simulate a realistic interview scenario, with messages exchanged in real-time.
