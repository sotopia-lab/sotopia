# ParallelSotopiaEnv Documentation

## Description
`ParallelSotopiaEnv` is a custom environment designed for parallel agents interacting within a defined simulation. This environment integrates various profiles, relationships, and actions that agents can perform, leveraging large language models (like "gpt-3.5-turbo") to model behaviors and interactions.

## Dependencies
```python
import asyncio
import copy
import itertools
import random
from typing import Any, Literal, Optional, Type, TypeVar
from gin import configurable
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.text import Text
from gymnasium.spaces import Space
from pettingzoo.utils.env import ParallelEnv
from redis_om.model.model import NotFoundError
from sotopia.agents.llm_agent import Agents
from sotopia.database import EnvironmentProfile
from sotopia.database.persistent_profile import AgentProfile, RelationshipType
from sotopia.messages import ActionType, AgentAction, MessengerMixin, Observation, ScriptBackground, SimpleMessage
from sotopia.renderers import RenderContext, XMLRenderer
from .evaluators import Evaluator, unweighted_aggregate_evaluate
```

## Class

### `ParallelSotopiaEnv`
```python
class ParallelSotopiaEnv(ParallelEnv[str, Observation, AgentAction], MessengerMixin):
```
#### Parameters
- `available_action_types` (set[ActionType], optional): The action types available to the agents. Defaults to `{"none", "speak", "non-verbal communication", "action", "leave"}`.
- `action_order` (Literal["simultaneous", "round-robin", "random"], optional): How agents take actions. Defaults to `"simultaneous"`.
- `model_name` (str, optional): Name of the language model. Defaults to `"gpt-3.5-turbo"`.
- `evaluators` (list[Evaluator], optional): List of evaluators for responses. Defaults to `[]`.
- `terminal_evaluators` (list[Evaluator], optional): List of evaluators for terminal states. Defaults to `[]`.
- `uuid_str` (str | None, optional): UUID to load the environment profile from the database.
- `env_profile` (EnvironmentProfile | None, optional): Profile of the environment.
- `background_class` (Optional[Type[TBackground]], optional): Class for background scenarios, defaults to `ScriptBackground`.

#### Usage Example
```python
env = ParallelSotopiaEnv(
    available_action_types={"none", "speak", "action"},
    action_order="round-robin",
    model_name="gpt-3.5-turbo"
)
```

### Methods

#### `reset`
```python
def reset(
    self,
    seed: int | None = None,
    options: dict[str, str] | None = None,
    agents: Agents | None = None,
    omniscient: bool = False,
    lite: bool = False
) -> dict[str, Observation]
```
Initializes a new episode.

##### Parameters
- `seed` (int, optional): Seed for random number generator.
- `options` (dict[str, str], optional): Additional options.
- `agents` (Agents, optional): Agent profiles for the environment.
- `omniscient` (bool, optional): If agents know the other agent's goals. Defaults to `False`.
- `lite` (bool, optional): If `True`, background information is minimal.

##### Returns
- `dict[str, Observation]`: Initial observations for each agent.

#### Usage Example
```python
observations = env.reset(
    seed=42,
    agents=agents,
    omniscient=True
)
```

#### `step`
```python
def step(
    self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
) -> tuple[
    dict[str, Observation],
    dict[str, float],
    dict[str, bool],
    dict[str, bool],
    dict[str, dict[Any, Any]]
]
```
Executes actions and returns new states. Observations are filtered per-agent based on private message visibility.

##### Parameters
- `actions` (dict[str, AgentAction] | dict[str, dict[str, int | str]]): Actions taken by agents. Each action can be:
  - An `AgentAction` object
  - A dictionary with `action_type` (string literal like `"speak"`, `"none"`, etc.) and `argument` (string)
  - Optionally includes a `to` field (list of strings) for private messages

##### Returns
- `tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]]
    ]`: Next state information, including observations, rewards, terminals, truncations, and additional info.

##### Private Message Visibility
- **Public actions** (no `to` field or `to=None`): Visible to all agents in their observations
- **Private actions** (with `to` field): Only visible to the sender and agents listed in `to`
- Each agent receives a filtered observation containing only actions they can see

#### Usage Example
```python
next_obs, rewards, done, truncations, info = env.step(actions)
```

#### `astep`
```python
async def astep(
    self, actions: dict[str, AgentAction] | dict[str, dict[str, int | str]]
) -> tuple[
    dict[str, Observation],
    dict[str, float],
    dict[str, bool],
    dict[str, bool],
    dict[str, dict[Any, Any]]
]
```
Asynchronous version of `step`. Observations are filtered per-agent based on private message visibility.

##### Parameters
- `actions` (dict[str, AgentAction] | dict[str, dict[str, int | str]]): Actions taken by agents. Each action can be:
  - An `AgentAction` object
  - A dictionary with `action_type` (string literal like `"speak"`, `"none"`, etc.) and `argument` (string)
  - Optionally includes a `to` field (list of strings) for private messages

##### Returns
- `tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]]
    ]`: Next state information, including observations, rewards, terminals, truncations, and additional info.

##### Private Message Visibility
- **Public actions** (no `to` field or `to=None`): Visible to all agents in their observations
- **Private actions** (with `to` field): Only visible to the sender and agents listed in `to`
- Each agent receives a filtered observation containing only actions they can see

#### Usage Example
```python
next_obs, rewards, done, truncations, info = await env.astep(actions)
```

#### `render`
```python
def render(self, mode: str = "human") -> None
```
Render the environment (not implemented).

#### `close`
```python
def close(self) -> None
```
Close the environment (not implemented).

---

## Utility Functions

### `_actions_to_natural_language_for_viewer`
```python
def _actions_to_natural_language_for_viewer(
    actions: dict[str, AgentAction], viewer: str
) -> str
```
Converts agent actions to human-readable language, filtered for a specific viewer. Private messages are only included if the viewer is the sender or a recipient.

### `_map_gender_to_adj`
```python
def _map_gender_to_adj(gender: str) -> str
```
Maps gender to its adjective equivalent.

### `get_bio`
```python
def get_bio(
    relationship: RelationshipType, profile: AgentProfile, agent_id: int
) -> str
```
Generates a description of the agent based on their relationship type.

#### Parameters
- `relationship` (RelationshipType): Type of relationship.
- `profile` (AgentProfile): Agent profile information.
- `agent_id` (int): Agent identifier.

#### Returns
- `str`: Rendered biography string for the profile.

### `render_text_for_agent`
```python
@configurable
def render_text_for_agent(
    raw_text: str,
    agent_id: int,
    tags_to_render: list[str] = [
        "extra_info",
        "clarification_hint",
        "strategy_hint",
    ],
) -> str
```
Renders text viewable by a specific agent using XMLRenderer.

---

## Action Space

The action space for each agent is a `Dict` space with:
- `action_type`: A `LiteralSpace` that samples string literals (e.g., `"speak"`, `"none"`, `"action"`) from `available_action_types`
- `argument`: A `Text` space (max 256 characters) for the action content

**Note**: The `action_type` is now a string literal, not an integer index. When sampling from the action space, you'll get strings like `"speak"` instead of integers like `0`.

## Usage Example
Here's a typical usage example starting an episode in the environment:

```python
# Initialize environment
env = ParallelSotopiaEnv(
    available_action_types={"none", "speak", "non-verbal communication", "action"},
    action_order="simultaneous"
)

# Reset the environment
observations = env.reset(
    seed=42,
    agents={
        "agent_1": Agent(...),
        "agent_2": Agent(...),
        "agent_3": Agent(...)
    },
    omniscient=True
)

# Perform public actions (visible to all)
actions = {
    "agent_1": AgentAction(action_type="speak", argument="Hello everyone!"),
    "agent_2": AgentAction(action_type="action", argument="waved"),
    "agent_3": AgentAction(action_type="speak", argument="Hi there!"),
}

next_obs, rewards, done, truncations, info = env.step(actions)

# Perform actions with private messages
actions_with_private = {
    "agent_1": AgentAction(
        action_type="speak",
        argument="Psst, agent_2, let's discuss this privately",
        to=["agent_2"]  # Only visible to agent_1 and agent_2
    ),
    "agent_2": AgentAction(action_type="speak", argument="Hello everyone!"),  # Public
    "agent_3": AgentAction(
        action_type="speak",
        argument="I'll talk to agent_1",
        to=["agent_1"]  # Only visible to agent_1 and agent_3
    ),
}

next_obs, rewards, done, truncations, info = env.step(actions_with_private)

# Check observations - each agent sees different things
print("Agent 1 sees:", next_obs["agent_1"].last_turn)
# Includes both private messages (from agent_1 and agent_3) and public message from agent_2

print("Agent 2 sees:", next_obs["agent_2"].last_turn)
# Includes private message from agent_1 and public message from agent_2

print("Agent 3 sees:", next_obs["agent_3"].last_turn)
# Includes private message from agent_3 and public message from agent_2
# Does NOT include the private message from agent_1 to agent_2
```
