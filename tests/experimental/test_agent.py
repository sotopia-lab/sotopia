import asyncio

from aact import Message
from sotopia.experimental import BaseAgent

from aact.messages import Tick

from redis.asyncio import Redis

import pytest


class ReturnPlusOneAgent(BaseAgent[Tick, Tick]):
    async def aact(self, observation: Tick) -> Tick:
        print(observation)
        return Tick(tick=observation.tick + 1)


@pytest.mark.asyncio
async def test_base_agent() -> None:
    async with ReturnPlusOneAgent(
        node_name="test_base_agent",
        input_channel_types=[("input", Tick)],
        output_channel_types=[("output", Tick)],
        redis_url="redis://localhost:6379/0",
    ) as agent1:
        async with ReturnPlusOneAgent(
            node_name="test_base_agent_2",
            input_channel_types=[("output", Tick)],
            output_channel_types=[("final", Tick)],
            redis_url="redis://localhost:6379/0",
        ) as agent2:
            try:
                await super(ReturnPlusOneAgent, agent1).aact(Tick(tick=1))
                assert False, "Should raise NotImplementedError"
            except NotImplementedError:
                pass

            try:
                async for _ in agent1.event_handler(
                    "output", Message[Tick](data=Tick(tick=2))
                ):
                    pass
                assert False, "Should raise ValueError"
            except ValueError:
                pass

            redis = Redis()
            r = redis.pubsub()
            await r.subscribe("final")

            task_agent_1 = asyncio.create_task(agent1.event_loop())
            task_agent_2 = asyncio.create_task(agent2.event_loop())

            await redis.publish(
                "input", Message[Tick](data=Tick(tick=1)).model_dump_json()
            )

            async def _() -> None:
                async for message in r.listen():
                    if message["type"] == "message":
                        assert message["channel"] == b"final"
                        assert message["data"] == Message[Tick](
                            data=Tick(tick=3)
                        ).model_dump_json().encode("utf-8")
                        return

            try:
                await asyncio.wait_for(_(), timeout=1)
            finally:
                task_agent_1.cancel()
                task_agent_2.cancel()
                await r.unsubscribe("final")
                await redis.close()
