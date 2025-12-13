# Python Classes for Messages

## Overview
This module provides several classes to represent scripted interactions between participants in a conversation. The key classes are `Message`, `SimpleMessage`, `Observation`, `ScriptBackground`, `ScriptEnvironmentResponse`, `AgentAction`, and `ScriptInteraction`.

## Classes

### `Message`

An interface for messages. There is only one required method:

#### Methods

- `to_natural_language(self) -> str`: Raises `NotImplementedError`.

### `SimpleMessage`

A simple message with a single string field.

#### Attributes

- `message: str`: The message content.

#### Methods

- `to_natural_language(self) -> str`: Returns the message as a string.

### `Observation`

Represents an observation in a conversation.

#### Attributes

- `last_turn: str`: The last turn of the conversation.
- `turn_number: int`: The turn number of the conversation.
- `available_actions: list[ActionType]`: The available actions.

#### Methods

- `to_natural_language(self) -> str`: Returns a string describing the conversation turn.

### `ScriptBackground`

Contains the background information for a scripted episode.

#### Attributes

- `scenario: str`: The scenario of the episode.
- `p1_name: str`: The name of participant 1.
- `p2_name: str`: The name of participant 2.
- `p1_background: str`: The background of participant 1.
- `p2_background: str`: The background of participant 2.
- `p1_goal: str`: The goal of participant 1.
- `p2_goal: str`: The goal of participant 2.

#### Methods

- `to_natural_language(self) -> str`: Returns a string describing the episode context.

### `ScriptEnvironmentResponse`

Represents the environment's response to the interaction.

#### Attributes

- `terminated: bool`: Whether the conversation is terminated.
- `p1_rate: float | tuple[float, dict[str, float]] | None`: Rating of participant 1.
- `p2_rate: float | tuple[float, dict[str, float]] | None`: Rating of participant 2.
- `comments: str | None`: Supporting comments for the termination and rating.

#### Methods

- `to_natural_language(self) -> str`: Returns a string describing the environment's response.

### `AgentAction`

Represents an action taken by an agent. Actions can be either public (visible to all agents) or private (visible only to specific recipients).

#### Attributes

- `action_type: ActionType`: The type of action. Can be one of: `"none"`, `"speak"`, `"non-verbal communication"`, `"action"`, or `"leave"`.
- `argument: str`: The argument associated with the action (e.g., the utterance for `"speak"`, the description for `"action"`).
- `to: list[str] | None`: (Optional) List of recipient agent names. When specified, the action is a private message visible only to the sender and the listed recipients. When `None` or empty, the action is public and visible to all agents. Defaults to `None`.

#### Methods

- `to_natural_language(self) -> str`: Returns a string describing the agent's action. Private messages are prefixed with `[private to {recipients}]`.

#### Private Messages

Private messages allow agents to communicate privately with specific recipients. When an action has a `to` field specified:

- The action is only visible to the sender and the agents listed in `to`
- Other agents will not see the action in their observations
- The `to` field is validated to ensure recipients are valid agent names and the sender cannot target themselves

#### Validation

The `to` field is validated when creating an `AgentAction` with context:
- Recipients must be valid agent names in the environment
- Senders cannot send private messages to themselves
- Invalid recipients will raise a `ValueError` with details about allowed recipients

### `ScriptInteraction`

Represents the entire interaction between participants.

#### Attributes

- `interactions: str`: The script of the interaction.

#### Methods

- `to_natural_language(self) -> str`: Returns the raw interaction script.
- `parse(self, agent_names: list[str], background: str) -> tuple[list[list[tuple[str, str, Message]]], list[tuple[str, Message]]]`: Parses the interaction script.
- `parse_single_dialogue(self, dialogue: str) -> dict[str, str | int | AgentAction | None]`: Parses a single dialogue line.
- `split_by_turn(self, input_string: str) -> list[str]`: Splits the input script by turns.
- `default_value_for_return_type() -> ScriptInteractionReturnType`: Provides a default return type value.

## Usage Example

```python
# Create instances of different message types
simple_msg = SimpleMessage(message="Hello, world!")
print(simple_msg.to_natural_language())

observation = Observation(
    last_turn="Hi there!",
    turn_number=1,
    available_actions=["speak", "none"]
)
print(observation.to_natural_language())

script_background = ScriptBackground(
    scenario="Business Meeting",
    p1_name="Alice",
    p2_name="Bob",
    p1_background="Engineer",
    p2_background="Manager",
    p1_goal="Discuss project",
    p2_goal="Approve budget"
)
print(script_background.to_natural_language())

response = ScriptEnvironmentResponse(
    terminated=True,
    p1_rate=9.5,
    p2_rate=7.0,
    comments="A lively discussion."
)
print(response.to_natural_language())

# Public action (visible to all agents)
action = AgentAction(action_type="speak", argument="Hello, how can I help you?")
print(action.to_natural_language())
# Output: said: "Hello, how can I help you?"

# Private action (visible only to sender and specified recipients)
private_action = AgentAction(
    action_type="speak",
    argument="Psst, let's discuss this privately",
    to=["agent2", "agent3"]
)
print(private_action.to_natural_language())
# Output: [private to ['agent2', 'agent3']] said: "Psst, let's discuss this privately"

interaction_script = """
Turn #1
Alice said: "Hello, Bob!"

Turn #2
Bob said: "Hi, Alice! How's the project going?"
"""
script_interaction = ScriptInteraction(interactions=interaction_script)
print(script_interaction.to_natural_language())
```
