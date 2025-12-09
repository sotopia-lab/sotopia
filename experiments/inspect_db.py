import asyncio
from typing import cast
from sotopia.database.logs import EpisodeLog


async def inspect_db(tag: str, limit: int = 1) -> None:
    print(f"Inspecting DB for tag: {tag}")
    try:
        episodes = cast(list[EpisodeLog], EpisodeLog.find(EpisodeLog.tag == tag).all())
        if not episodes:
            print("No episodes found.")
            return

        print(f"Found {len(episodes)} episodes. Showing last {limit}:")

        for ep in episodes[-limit:]:
            print(f"\n--- Episode PK: {ep.pk} ---")
            print(f"Environment: {ep.environment}")
            print(f"Models: {ep.models}")
            print(f"Rewards: {ep.rewards}")
            print(f"Reasoning: {ep.reasoning}")

            # Print messages summary
            print("Messages (first 3):")
            for turn in ep.messages[:1]:  # Check first turn
                for sender, type_str, content in turn:
                    print(f"  [{sender}] ({type_str}): {content[:100]}...")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import os
    import sys

    os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")
    tag = sys.argv[1] if len(sys.argv) > 1 else "elo_exp_v1"
    asyncio.run(inspect_db(tag))
