# Multi-Agent Tests

This directory contains test scenarios for Sotopia's multi-agent (3+ agents)
with private action support.

## Features

- Private actions

    Different from private actions, private actions are only visible to the author
    and the recipients but not others.

    Agents can set the `to` field in `AgentAction` class to specify the
    recipients.

    It is called private *actions* as the `to` field is not limited to the
    "speak" action. Any action, e.g. "non-verbal communication", "action", or
    even "leave" can have such property.

## Prerequisites

- Redis-stack installed and running

    From the root of this project, run:

    ```sh
    mkdir -p examples/experimental/multi_agents_private_dm/redis-data
    redis-stack-server --dir examples/experimental/multi_agents_private_dm/redis-data
    ```

- OpenAI API key (if not using local models)
- Python 3.11+ with sotopia installed

## Running Multi-Agent Tests

### 2 Agent Are Having a Private Conversation, 1 Agent Passing By

Run the script with

```sh
uv run examples/experimental/multi_agents_private_dm/multi_agents_private_dm.py
```

#### Scenario

> Two of the agents (Alice and Bob) are friends, and one stranger (Charlie).
>
> The friends are talk to each other privately and the stranger does not know
> what they are saying.

#### Expected Behavior

1. The fiends are having a private conversation with each other, completely
   ignore the stranger, Charlie.
2. Charlie knows nothing about the content of conversation throughout the
   simulation.

## Technical Details

- Added a **new field to support private actions**: `AgentAction.to: list[str]
  | None`. When set, the action is visible only to the sender and listed
  recipients; otherwise it is public.

- **Per-viewer** observations: `_actions_to_natural_language_for_viewer(actions,
  viewer)` filters actions so each agent only sees public actions plus private
  ones they sent or are addressed in.

- `MessengerMixin` adds `private_inbox` and to separate public and private
  transcripts. Env writes private actions with `private=True`.

- Prompt updates: action-generation prompts in `generation_utils/generate.py`
  instruct models to set `to` for private actions and clarify that public
  actions omit it.
