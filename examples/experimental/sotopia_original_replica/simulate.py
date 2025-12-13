import asyncio
import argparse
import json
import os
from typing import Dict, Any, Optional
from sotopia.experimental.server import arun_one_episode


async def simulate_from_config(
    episode_config: Dict[str, Any], save_path: Optional[str] = None
) -> None:
    # Generate a unique connection ID
    connection_id = ""  # Empty string for non-streaming mode

    # Call the updated arun_one_episode with the episode_config
    result = None
    async for message in arun_one_episode(episode_config, connection_id):
        # Store the last message as the result
        result = message

    # If save_path is provided, save the result to a file
    if save_path and result:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Sotopia simulation")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--save_path", type=str, help="Path to save the result")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            episode_config = json.load(f)
    else:
        # Default configuration if no config file is provided
        agent_ids = ["Jack", "Jill", "John"]
        # Create a default scenario
        scenario = "Just chat (finish the conversation in 2 turns)"
        agent_goals = ["Just chat"] * len(agent_ids)

        # Create the episode config directly
        episode_config = {
            "redis_url": "redis://localhost:6379/0",
            "extra_modules": [
                "examples.experimental.sotopia_original_replica.llm_agent_sotopia",
                "sotopia.experimental.agents.redis_agent",
            ],
            "agent_node": "llm_agent",
            "default_model": "gpt-4o-mini",
            "evaluator_model": "gpt-4o",
            "use_pk_value": False,
            "push_to_db": False,
            "evaluate_episode": False,
            "max_turns": 20,
            "scenario": scenario,
            "agents": [
                {
                    "name": agent_id,
                    "goal": agent_goals[i],
                    "model_name": "gpt-4o-mini",
                    "background": {
                        "pk": agent_id,
                        "first_name": agent_id,
                        "last_name": agent_id,
                        "model": "gpt-4o-mini",
                    },
                }
                for i, agent_id in enumerate(agent_ids)
            ],
        }
    if not args.save_path:
        args.save_path = "./data/sotopia_original_replica_test.json"
    asyncio.run(simulate_from_config(episode_config, args.save_path))
    print(f"Simulation completed. Result saved to {args.save_path}")


if __name__ == "__main__":
    main()
