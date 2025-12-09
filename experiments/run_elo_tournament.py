import asyncio
import logging
import random
import os
import sys
import itertools
from copy import deepcopy
from datetime import datetime

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sotopia.server import arun_one_episode
from examples.experimental.werewolves.main import (  # type: ignore[import-untyped]
    prepare_scenario,
    load_config,
    CONFIG_PATH,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
# Suppress verbose logs from sotopia and litellm
logging.getLogger("sotopia.generation").setLevel(logging.INFO)
# logging.getLogger("LiteLLM").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


async def run_elo_tournament(
    env_name: str,
    models: list[str],
    n_episodes: int = 10,
    tag: str = "elo_exp_v1",
    push_to_db: bool = True,
) -> None:
    """
    Run a tournament with randomized roles and custom Werewolf logic.
    """
    print(f"Starting ELO Tournament: {env_name}")
    print(f"Competitors: {models}")
    print(f"Episodes: {n_episodes}")

    # Load base config once
    base_config = load_config(CONFIG_PATH)

    # Get base agent list and role goals
    base_agents = base_config.get("agents", [])
    roles = [a["role"] for a in base_agents]

    # Name Pool
    NAME_POOL = [
        "James",
        "Mary",
        "Robert",
        "Patricia",
        "John",
        "Jennifer",
        "Michael",
        "Linda",
        "David",
        "Elizabeth",
        "William",
        "Barbara",
        "Richard",
        "Susan",
        "Joseph",
        "Jessica",
        "Thomas",
        "Sarah",
        "Charles",
        "Karen",
        "Christopher",
        "Nancy",
        "Daniel",
        "Lisa",
        "Matthew",
        "Betty",
        "Anthony",
        "Margaret",
        "Mark",
        "Sandra",
        "Donald",
        "Ashley",
        "Steven",
        "Kimberly",
        "Paul",
        "Emily",
        "Andrew",
        "Donna",
        "Joshua",
        "Michelle",
        "Kenneth",
        "Dorothy",
        "Kevin",
        "Carol",
        "Brian",
        "Amanda",
        "George",
        "Melissa",
        "Edward",
        "Deborah",
        "Ronald",
        "Stephanie",
        "Timothy",
        "Rebecca",
        "Jason",
        "Sharon",
        "Jeffrey",
        "Laura",
        "Ryan",
        "Cynthia",
        "Jacob",
        "Kathleen",
        "Gary",
        "Amy",
        "Nicholas",
        "Shirley",
        "Eric",
        "Angela",
        "Jonathan",
        "Helen",
        "Stephen",
        "Anna",
        "Larry",
        "Brenda",
        "Justin",
        "Pamela",
        "Scott",
        "Nicole",
        "Brandon",
        "Emma",
    ]

    role_to_team = {
        "Werewolf": "Werewolves",
        "Villager": "Villagers",
        "Seer": "Villagers",
        "Witch": "Villagers",
    }
    # 1. Randomize Config
    if len(models) < 2:
        print("Error: Need at least 2 models for ELO tournament (A vs B).")
        return

    # Generate all unique pairs (5C2 = 10 pairs if 5 models)
    pairs = list(itertools.combinations(models, 2))
    print(
        f"Tournament Configurations: {len(pairs)} matchups based on {len(models)} models."
    )
    for idx, (m1, m2) in enumerate(pairs):
        print(f"Matchup {idx+1}/{len(pairs)}: {m1} vs {m2}")

    # Loop through each pair
    for pair_idx, (model_a, model_b) in enumerate(pairs):
        print(f"\nStarting Matchup {pair_idx+1}: {model_a} vs {model_b}")

        for i in range(n_episodes):
            print(f"  > Episode {i+1}/{n_episodes} for Matchup {pair_idx+1}")

            # Randomly naming and role assignment
            new_agents_config = []
            current_names = random.sample(NAME_POOL, len(roles))

            # Map roles to models for this episode
            # To be fair, we randomly swap who plays Werewolf vs Villager in each episode
            # OR we can keep it fixed if we want A=Werewolf vs B=Villager.
            # Given "5C2" usually implies "A vs B", testing both sides is better.
            # Let's randomly assign Team A (Werewolf) and Team B (Villager) from the pair.

            if random.random() < 0.5:
                model_werewolf, model_villager = model_a, model_b
            else:
                model_werewolf, model_villager = model_b, model_a

            # Construct config with randomized names
            for role_name, agent_name in zip(roles, current_names):
                team = role_to_team.get(
                    role_name, "Villagers"
                )  # Changed default to "Villagers" for consistency
                new_agents_config.append(
                    {"name": agent_name, "role": role_name, "team": team}
                )

            episode_config = deepcopy(base_config)
            episode_config["agents"] = new_agents_config

            episode_models = []
            for agent_cfg in new_agents_config:
                if agent_cfg["team"] == "Werewolves":
                    episode_models.append(model_werewolf)
                else:
                    episode_models.append(model_villager)

            print(f"    Roles: {[(a['name'], a['role']) for a in new_agents_config]}")
            print(
                f"    Models: Werewolves({model_werewolf}) vs Villagers({model_villager})"
            )

            # 3. Instantiate Game
            env, agents = prepare_scenario(
                env_model_name="gpt-4",
                agent_model_name=episode_models,
                config=episode_config,
            )

            # 4. Run Episode
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Include model names in filename for easier analysis
            # Sanitize model names for filename
            sanitized_m1 = (
                model_werewolf.replace("/", "_").replace("@", "_").split("v1")[0][-10:]
            )
            sanitized_m2 = (
                model_villager.replace("/", "_").replace("@", "_").split("v1")[0][-10:]
            )

            log_filename = f"episode_{tag}_match{pair_idx}_{sanitized_m1}_vs_{sanitized_m2}_{i}_{timestamp}.json"

            # ensure logs directory exists
            os.makedirs("logs", exist_ok=True)
            log_path = os.path.join("logs", log_filename)

            await arun_one_episode(
                env=env,
                agent_list=agents,
                tag=tag,
                push_to_db=push_to_db,
                output_path=log_path,
            )
            print(f"    Episode saved to {log_path}")

    print("Tournament Complete.")


if __name__ == "__main__":
    import os
    import argparse

    os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")

    parser = argparse.ArgumentParser(description="Run Werewolf ELO Tournament")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-5",
            "custom/google/gemma-3-1b@http://127.0.0.1:1234/v1",
            "custom/qwen/qwen3-next-80b@http://127.0.0.1:1234/v1",
            "custom/qwen/qwen3-4b-2507@http://127.0.0.1:1234/v1",
        ],
        help="List of models to compete",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to run"
    )
    parser.add_argument("--tag", type=str, default="elo_exp_v1", help="Experiment tag")

    args = parser.parse_args()

    asyncio.run(
        run_elo_tournament(
            env_name="Werewolves Game",
            models=args.models,
            n_episodes=args.episodes,
            tag=args.tag,
            push_to_db=True,
        )
    )
