import asyncio

from aact import Message
from sotopia.experimental.agents import BaseAgent

from aact.messages import Tick

from redis.asyncio import Redis

import pytest


class ReturnPlusOneAgent(BaseAgent[Tick, Tick]):
    async def aact(self, observation: Tick) -> Tick:
        print(observation)
        return Tick(tick=observation.tick + 1)


@pytest.mark.asyncio
async def test_return_plus_one_agent() -> None:
    async with ReturnPlusOneAgent(
        input_channel_types=[("input", Tick)],
        output_channel_types=[("output", Tick)],
        redis_url="redis://localhost:6379/0",
    ) as agent1:
        async with ReturnPlusOneAgent(
            input_channel_types=[("output", Tick)],
            output_channel_types=[("final", Tick)],
            redis_url="redis://localhost:6379/0",
        ) as agent2:
            redis = Redis()
            r = redis.pubsub()
            await r.subscribe("final")

            asyncio.create_task(agent1.event_loop())
            asyncio.create_task(agent2.event_loop())

            await redis.publish(
                "input", Message[Tick](data=Tick(tick=1)).model_dump_json()
            )

            async with asyncio.timeout(5):
                async for message in r.listen():
                    print(message)
                    if message["type"] == "message":
                        assert message["channel"] == b"final"
                        assert message["data"] == Message[Tick](
                            data=Tick(tick=3)
                        ).model_dump_json().encode("utf-8")
                        return

            assert False, "Timeout reached"
