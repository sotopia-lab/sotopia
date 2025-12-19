import asyncio
import json
import os
import sys

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from typing import Any, AsyncIterator
from aact import Message, Node, NodeFactory
from aact.messages import Audio
from websockets.asyncio.client import connect, ClientConnection

import base64

URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"


@NodeFactory.register("openai_realtime")
class OpenAIRealtimeNode(Node[Audio, Audio]):
    def __init__(
        self, input_channel: str, output_channel: str, instruction: str, redis_url: str
    ) -> None:
        super().__init__(
            input_channel_types=[(input_channel, Audio)],
            output_channel_types=[(output_channel, Audio)],
            redis_url=redis_url,
        )
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.instruction = instruction
        self.websocket: ClientConnection | None = None
        self.task: asyncio.Task[None] | None = None

    async def ws_listener(self) -> None:
        assert self.websocket is not None, "Websocket is not initialized"
        async for message in self.websocket:
            data = json.loads(message)
            if data["type"] == "response.audio.delta":
                delta = base64.b64decode(data["delta"])
                await self.r.publish(
                    self.output_channel,
                    Message[Audio](data=Audio(audio=delta)).model_dump_json(),
                )
            elif data["type"] == "error":
                print(data["error"]["message"])

    async def __aenter__(self) -> Self:
        self.websocket = await connect(
            URL,
            additional_headers={
                "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                "OpenAI-Beta": "realtime=v1",
            },
        )
        await self.websocket.__aenter__()
        await self.websocket.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": self.instruction}],
                    },
                }
            )
        )
        await self.websocket.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio", "text"],
                        "voice": "alloy" if self.output_channel == "Jane" else "echo",
                        "instructions": self.instruction,
                    },
                }
            )
        )
        self.task = asyncio.create_task(self.ws_listener())
        return await super().__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.websocket is not None:
            await self.websocket.__aexit__(exc_type, exc_value, traceback)
        if self.task is not None:
            self.task.cancel()
        await super().__aexit__(exc_type, exc_value, traceback)

    async def event_handler(
        self, channel: str, message: Message[Audio]
    ) -> AsyncIterator[tuple[str, Message[Audio]]]:
        if channel == self.input_channel:
            assert self.websocket is not None, "Websocket is not initialized"
            # await self.websocket.send(
            #     json.dumps(
            #         {
            #             "type": "conversation.item.create",
            #             "item": {
            #                 "type": "message",
            #                 "role": "user",
            #                 "content": [
            #                    {"type": "input_text", "text": message.data.text}
            #                 ],
            #             },
            #         }
            #     )
            # )
            # await self.websocket.send(json.dumps({"type": "response.create"}))
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(message.data.audio).decode(),
                    }
                )
            )

        else:
            raise ValueError(f"Unexpected channel: {channel}")
            yield "", Message(data=Audio(audio=b""))  # Unreachable code
