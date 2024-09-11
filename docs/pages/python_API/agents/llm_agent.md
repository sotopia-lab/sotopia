# `llm_agent.py`

This Python module includes several classes and utilities for implementing agents within an environment using asynchronous execution (`asyncio`). The key class, `LLMAgent`, leverages a large language model (LLM) to generate actions based on observations and goals. It utilizes the `sotopia` library, presumably for multi-agent systems, and can produce script-like outputs if specified. Other classes include `HumanAgent` for human input and `ScriptWritingAgent` for generating scripted actions.

## Classes

### `LLMAgent`

This class represents an agent that interacts with an environment using a language model (LLM) to generate actions based on observations and goals.

#### Constructor: `__init__`

Initializes an instance with several attributes:

**Arguments:**
- `agent_name`: `str | None`, default `None`. Used for agent identification.
- `uuid_str`: `str | None`, default `None`. A unique identifier for the agent.
- `agent_profile`: `AgentProfile | None`, default `None`. An optional profile for the agent.
- `model_name`: `str`, default `"gpt-3.5-turbo"`. Specifies the LLM model used.
- `script_like`: `bool`, default `False`. Determines if the agent produces script-like outputs.

**Usage:**
```python
agent = LLMAgent(agent_name="My Agent", model_name="gpt-3.5-turbo")
```

**Attributes:**
- `model_name`: `str | None`. The name of the LLM model, defaulting to `"gpt-3.5-turbo"`.
- `script_like`: `bool`. Determines if the agent produces script-like output.

#### Methods

- `goal()`: Returns the agent's goal as a string. If not set, it raises an exception.

  **Returns:** `str`

  **Usage:**
  ```python
  llm = LLMAgent()
  llm.goal = "Write a poem about nature"
  ```

- `act()`: Generates an action based on the agent's observations and goals.

  **Arguments:**
  - `_obs`: `Observation`. The current observation from the environment.

  **Returns:** `AgentAction`

  **Usage:**
  ```python
  agent = LLMAgent()
  action = agent.act(observation)
  ```

### `ScriptWritingAgent`

This class extends the `LLMAgent` to generate actions by composing scripts based on observations using a language model.

#### Constructor: `__init__`

**Arguments:**
- `agent_name`: `str | None`, default `None`.
- `uuid_str`: `str | None`, default `None`.
- `agent_profile`: `AgentProfile | None`, default `None`.
- `model_name`: `str`, default `"gpt-3.5-turbo"`.
- `background`: `ScriptBackground`, required for initialization.

**Usage:**
```python
script = ScriptWritingAgent(agent_name="User", background=ScriptBackground())
```

**Attributes:**
- `model_name`: `str | None`, default `"gpt-3.5-turbo"`.
- `agent_names`: `list[str]`, stores agent names.
- `background`: `ScriptBackground`, required for script generation.

### `HumanAgent`

This class simulates a human agent interacting with the environment by prompting the user for input and acting based on the user's choices.

#### Constructor: `__init__`

**Arguments:**
- `agent_name`: `str | None`, default `None`.
- `uuid_str`: `str | None`, default `None`.
- `agent_profile`: `AgentProfile | None`, default `None`.

**Usage:**
```python
agent = HumanAgent(agent_name="User", uuid_str=None)
```

**Attributes:**
- `model_name`: `"human"`. Specifies that this agent represents a human user.

#### Methods

- `goal()`: Returns the current goal, or prompts the user for input if no goal is set.

  **Usage:**
  ```python
  human_agent = HumanAgent()
  goal = human_agent.goal
  ```

- `act()`: Prompts the user to select an action based on observations and returns an `AgentAction`.

  **Usage:**
  ```python
  human_agent = HumanAgent()
  human_agent.act(obs={"available_actions": ["move", "jump"]})
  ```

### `Agents`

This class manages a collection of agents and allows for resetting or performing actions across multiple agents.

#### Methods

- `reset()`: Resets all agents in the collection.

  **Usage:**
  ```python
  agents = Agents({"agent1": Agent(), "agent2": Agent()})
  agents.reset()
  ```

- `act()`: Collects actions from all agents based on their respective observations.

  **Arguments:**
  - `obs`: `dict[str, Observation]`

  **Returns:** `dict[str, AgentAction]`

  **Usage:**
  ```python
  agents = Agents()
  actions = agents.act(obs_dict)
  ```

## Functions

### `ainput()`

Asynchronously retrieves user input while other tasks run concurrently.

**Arguments:**
- `prompt`: `str`, default `""`.

**Returns:** `str`

**Usage:**
```python
async def main():
    response = await ainput("Enter your name: ")
    task_name = response
```
