# AppWorld Agent Example

In this example, we explore the interaction between two friends, "Harry" and "Ron", who are planning a roadtrip from Seattle to Pittsburgh.
These agents are designed to interact with mocked APIs provided by AppWorld.

## Agents Overview

- **Harry**: An LLM agent tasked to plan a roadtrip with Ron. His goal is to minimize distractions during the roadtrip.
- **Ron**: An LLM agent tasked to plan a roadtrip with Harry. His goal is to explore different historical sites as much as possible.

### Actions Available to LLM Agents

- **speak**: Communicate with other agents by sending a message.
- **none**: Choose not to take any action, often used when waiting for data.
- **non-verbal**: Perform a non-verbal action, such as a gesture.
- **choose_tool_action**: Choose which tool to use given all that's available in AppWorld.
- **tool_action**: Use a tool from AppWorld.
- **leave**: Exit the conversation when goals are completed or abandoned.

These actions enable the agents to interact dynamically with their environment and each other, providing a robust framework for simulating complex scenarios.

## Dataflow Configuration

The dataflow file `app_world_agent.toml` is used to configure the interaction between the agents. It acts as a blueprint, defining how the agents communicate and process information.

## Running the Example

### Prerequisites

1. **Redis**: Ensure Redis is hosted at `localhost:6379`. Update the `redis_url` in the `app_world_agent.toml` file if using a different Redis database.

2. **Setting up the AppWorld runtime**: First, you need to install appworld and run a Docker container that simulates all APIs within that environment.Our agent will call these APIs in a RESTful manner.

   ```bash
   pip install appworld
   appworld serve apis --docker
   ```

3. **Run the minimal example**: To run this example, we then use the `aact` tool:

   ```bash
   aact run-dataflow examples/experimental/appworld_agent/appworld_agent.toml
   ```
