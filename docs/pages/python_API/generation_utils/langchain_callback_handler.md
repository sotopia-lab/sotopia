## LoggingCallbackHandler

A `LoggingCallbackHandler` class that extends `StdOutCallbackHandler` to provide additional logging capabilities. This handler logs specific events during the chain execution process and captures prompts for further retrieval.

### Attributes

- `always_verbose`: A class attribute set to `True` to ensure verbose logging.
- `logger`: Logger instance initialized with the provided name.
- `prompt`: Stores the prompt formatted during the chain execution.

### Methods

#### __init__(self, name: str) -> None

Initializes the `LoggingCallbackHandler`.

- **Parameters:**
  - `name` (`str`): The name of the logger.

#### on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any) -> None

Handle the start of a chat model, capturing and logging the first message's content.

- **Parameters:**
  - `serialized` (`dict[str, Any]`): Serialized chat model details.
  - `messages` (`list[list[BaseMessage]]`): List of messages in the chat model.
  - `kwargs` (`Any`): Additional keyword arguments.

#### on_chain_start(self, *args: Any, **kwargs: Any) -> None

Placeholder for handling the start of a chain.

#### on_chain_end(self, *args: Any, **kwargs: Any) -> None

Placeholder for handling the end of a chain.

#### on_agent_action(self, *args: Any, **kwargs: Any) -> Any

Placeholder for handling agent actions.

#### on_tool_end(self, *args: Any, **kwargs: Any) -> None

Placeholder for handling the end of a tool.

#### on_tool_error(self, error: BaseException | KeyboardInterrupt, **kwargs: Any) -> None

Handle tool errors by doing nothing.

- **Parameters:**
  - `error` (`BaseException | KeyboardInterrupt`): The error encountered.
  - `kwargs` (`Any`): Additional keyword arguments.

#### on_text(self, text: str, color: str | None = None, end: str = "", **kwargs: Any) -> None

Placeholder for handling text output when an agent ends.

- **Parameters:**
  - `text` (`str`): The text to output.
  - `color` (`str | None`): Optional color for the text.
  - `end` (`str`): Optional string to append at the end.
  - `kwargs` (`Any`): Additional keyword arguments.

#### retrieve_prompt(self) -> str

Retrieve the captured prompt.

- **Returns:**
  - `str`: The captured prompt.

#### on_agent_finish(self, *args: Any, **kwargs: Any) -> None

Placeholder for handling when an agent finishes.

### Usage Example

```python
import logging
from your_module import BaseMessage, LoggingCallbackHandler

# Configuration of logging
logging.basicConfig(level=15)

# Initialize the handler
handler = LoggingCallbackHandler(name="langchain")

# Example usage with mock data
serialized = {}
messages = [[BaseMessage(content="Hello, how can I assist you today?")]]

handler.on_chat_model_start(serialized=serialized, messages=messages)
print(handler.retrieve_prompt())  # Output: "Hello, how can I assist you today?"
```
