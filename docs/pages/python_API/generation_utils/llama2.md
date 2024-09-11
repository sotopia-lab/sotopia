## Llama2 Documentation

### Description

The `Llama2` class is a chat model based on Together's Llama-2-7b model. It extends the `BaseChatModel` from LangChain, and it is designed to generate responses based on input messages. The communication format follows the protocol defined by Together AI.

### Methods

#### `_convert_message_to_dict(message: BaseMessage) -> dict[str, Any]`

Converts a `BaseMessage` into a dictionary that includes the message's role and content.

##### Parameters

- `message` (`BaseMessage`): The input message to convert.

##### Returns

- `dict[str, Any]`: A dictionary representation of the message.

#### `_convert_dict_to_message(_dict: dict[str, str]) -> BaseMessage`

Converts a dictionary back into a `BaseMessage`.

##### Parameters

- `_dict` (`dict[str, str]`): The dictionary containing the message data.

##### Returns

- `BaseMessage`: The message reconstructed from the dictionary.

#### `_make_prompt_from_dict(dialog: List[dict[str, str]]) -> str`

Creates a prompt string from a dialog list, formatted according to Together AI's chat protocol.

##### Parameters

- `dialog` (`List[dict[str, str]]`): The dialog list to convert into a prompt string.

##### Returns

- `str`: The formatted prompt string.

### Class: `Llama2`

#### Attributes

- `client` (`type[together.Complete]`): Together API client.
- `model_name` (`str`): Name of the model to use.
- `temperature` (`float`): Sampling temperature for response generation.
- `max_tokens` (`int`): Maximum number of tokens to generate.
- `top_p` (`float`): Nucleus sampling parameter.
- `top_k` (`int`): Top-k sampling parameter.
- `repetition_penalty` (`float`): Penalty for repeating tokens.
- `start` (`bool`): Start flag.
- `_llm_type` (`str`): Internal type identifier.

#### Configuration

Uses Pydantic's configuration with extra parameters ignored (`Extra.ignore`).

#### `_generate(messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult`

Generates chat results based on input messages.

##### Parameters

- `messages` (`List[BaseMessage]`): List of input messages.
- `stop` (`Optional[List[str]]`): List of stop sequences.
- `run_manager` (`Optional[CallbackManagerForLLMRun]`): Run manager for callbacks.

##### Returns

- `ChatResult`: The chat result containing generated messages.

#### `_default_params() -> Dict[str, Any]`

Retrieves the default parameters for calling the Together API.

##### Returns

- `Dict[str, Any]`: The default parameters.

#### `_create_message_dicts(messages: List[BaseMessage], stop: Optional[List[str]]) -> Tuple[str, Dict[str, Any]]`

Creates message dictionaries and prompt strings from input messages.

##### Parameters

- `messages` (`List[BaseMessage]`): List of input messages.
- `stop` (`Optional[List[str]]`): List of stop sequences.

##### Returns

- `Tuple[str, Dict[str, Any]]`: The prompt string and associated parameters.

#### `_create_chat_result(response: Mapping[str, Any]) -> ChatResult`

Creates a `ChatResult` object from the API response.

##### Parameters

- `response` (`Mapping[str, Any]`): API response containing generated content.

##### Returns

- `ChatResult`: The result containing generated chat messages.

### Usage Example

```python
from langchain.schema import HumanMessage

# Initialize the Llama2 model
llama2_model = Llama2()

# Prepare input messages
messages = [HumanMessage(content="List the best restaurants in SF")]

# Generate a response
chat_result = llama2_model._generate(messages)

# Retrieve and print the response
response_message = chat_result.generations[0].message
print(response_message.content)
```

This documentation provides a concise overview and usage example for the `Llama2` class and its associated methods.
