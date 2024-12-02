import asyncio
import rich
from sotopia.generation_utils.generate import agenerate_character_profile

if __name__ == "__main__":
    agent_profile = asyncio.run(
        agenerate_character_profile(
            model_name="gpt-4o",
            basic_info="She likes certain brands for her clothes and shoes, and barely purchase items from other brands she is not familiar with. However, she likes to browse the shopping website from time to time, and only purchase some items when they are on sale. She is an extraverted person and asks a lot of questions in r/AskReddit regarding her own life or thought. She sometimes create new forums about a new topic that she is particularly interested in and welcome people joining. She has a Jeep Wrangler and likes to travel around, especially off-road. She often do road trips during vacations, and likes to plan for optimal travel routime in terms of time and distance.",
        )
    )
    rich.print(agent_profile)
