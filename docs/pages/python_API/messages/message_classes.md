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

Represents an action taken by an agent.

#### Attributes

- `action_type: ActionType`: The type of action.
- `argument: str`: The argument associated with the action.

#### Methods

- `to_natural_language(self) -> str`: Returns a string describing the agent's action.

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

action = AgentAction(action_type="speak", argument="Hello, how can I help you?")
print(action.to_natural_language())

interaction_script = """
Turn #1
Alice said: "Hello, Bob!"

Turn #2
Bob said: "Hi, Alice! How's the project going?"
"""
script_interaction = ScriptInteraction(interactions=interaction_script)
print(script_interaction.to_natural_language())
```
