import json
import os
from websockets.asyncio.client import connect

from pyaudio import PyAudio
import pyaudio

import base64

URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"


async def main() -> None:
    audio = PyAudio()

    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    if "OPENAI_API_KEY" not in os.environ:
        raise Exception("OPENAI_API_KEY is not set")
    else:
        async with connect(
            URL,
            additional_headers={
                "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                "OpenAI-Beta": "realtime=v1",
            },
        ) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {
                            "modalities": ["audio", "text"],
                            "instructions": "Write a short story about a robot that is trying to learn how to play chess.",
                        },
                    }
                )
            )

            async for message in websocket:
                data = json.loads(message)
                if data["type"] == "response.audio.delta":
                    delta = base64.b64decode(data["delta"])
                    stream.write(delta)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
