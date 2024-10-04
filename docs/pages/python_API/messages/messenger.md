# MessengerMixin Documentation

## Description
`MessengerMixin` is a mixin class that provides basic functionalities to handle an inbox for received messages. It allows for resetting the inbox and receiving new messages from specified sources.

## Methods

### `__init__() -> None`
Initializes an instance of `MessengerMixin` with an empty inbox.

#### Parameters
- None

#### Return Value
- None

### `reset_inbox() -> None`
Clears all messages from the inbox.

#### Parameters
- None

#### Return Value
- None

### `recv_message(source: str, message: Message) -> None`
Receives a new message from a specified source and adds it to the inbox.

#### Parameters
- `source` (str): The source of the incoming message.
- `message` (Message): The message object to be received.

#### Return Value
- None

## Attributes

### `inbox`
A list of tuples where each tuple contains the source (str) and the message (Message).

## Usage Example

```python
from .message_classes import Message
from .your_module import MessengerMixin

# Create an instance of MessengerMixin
messenger = MessengerMixin()

# Create a message object (assuming Message class is defined)
message = Message("Hello, World!")

# Receive a message from 'Alice'
messenger.recv_message("Alice", message)

# Check the current inbox
print(messenger.inbox)  # Output: [('Alice', <Message object>)]

# Reset the inbox
messenger.reset_inbox()

# Check the inbox again
print(messenger.inbox)  # Output: []
```
