import asyncio
import logging
import os
import sys
import json
import glob
from datetime import datetime
from tqdm.asyncio import tqdm

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sotopia.server import arun_one_episode
from experiments.utils import get_game_module, load_game_config

# Logger for this script
logger = logging.getLogger(__name__)


async def run_elo_tournament(
    game_names: list[str],
    tag: str = "elo_exp_v1",
    push_to_db: bool = True,
    concurrency_limit: int = 10,
) -> None:
    """
    Run ELO tournament by executing pre-generated rosters found in experiments/rosters/.
    """
    if not isinstance(game_names, list):
        game_names = [game_names]

    print("Starting ELO Tournament Execution")
    print(f"Target Games: {game_names}")
    print(f"Concurrency: {concurrency_limit}")
    print("\n" + "=" * 50)

    semaphore = asyncio.Semaphore(concurrency_limit)  # Limit concurrent API calls

    # Refactor: Define worker properly to avoid closure issues if defined once
    # Better to define it inside the loop or pass args.

    for game_name in game_names:
        print(f"Scanning rosters for game: {game_name}")

        # 1. Scan EXISTING LOGS to identify played rosters
        # We look for metadata["roster_file"] in the logs
        executed_rosters = set()
        log_pattern = os.path.join("logs", "*.json")
        existing_logs = glob.glob(log_pattern)

        # Optimization: Only scan logs relevant to this game?
        # But filename parsing is fast enough for <2000 files.
        for log_file in existing_logs:
            try:
                with open(log_file, "r") as f:
                    # Partial read might be faster but JSON load is safe
                    data = json.load(f)
                    meta = data.get("metadata", {})
                    r_file = meta.get("roster_file")
                    if r_file:
                        executed_rosters.add(r_file)
            except Exception as e:
                print(f"Error reading log file {log_file}: {e}")
                continue

        print(
            f"  Found {len(executed_rosters)} already executed rosters (across all games)."
        )

        roster_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "rosters", game_name)
        )
        roster_files = sorted(glob.glob(os.path.join(roster_dir, "*.json")))

        if not roster_files:
            print(f"  No rosters found in {roster_dir}")
            continue

        print(f"  Found {len(roster_files)} total rosters.")

        # Filter out executed rosters
        rosters_to_run = []
        for r_path in roster_files:
            r_filename = os.path.basename(r_path)
            if r_filename in executed_rosters:
                continue
            rosters_to_run.append(r_path)

        print(f"  {len(rosters_to_run)} rosters remaining to execute.")

        if not rosters_to_run:
            continue

        # Load Game Module
        try:
            game_module = get_game_module(game_name)
            prepare_scenario = game_module.prepare_scenario
        except Exception as e:
            print(f"  Error loading game module: {e}")
            continue

        async def _worker(roster_path: str) -> None:
            async with semaphore:
                filename = os.path.basename(roster_path)
                # Double check to prevent race condition if logs updated mid-run?
                # Not strictly necessary for this scale.

                try:
                    with open(roster_path, "r") as f:
                        roster_config = json.load(f)

                    base_config = load_game_config(game_name)
                    episode_config = base_config.copy()
                    episode_config.update(roster_config)

                    agent_model_list = [
                        a["agent_model"] for a in episode_config["agents"]
                    ]
                    parts = filename.replace(".json", "").split("_")
                    match_str = "unknown"
                    for p in parts:
                        if p.startswith("match"):
                            match_str = p

                    agents_conf = episode_config["agents"]
                    teams = set(a.get("team") for a in agents_conf if a.get("team"))

                    unique_models = sorted(list(set(agent_model_list)))
                    model_a_log = (
                        unique_models[0] if len(unique_models) > 0 else "unknown"
                    )
                    model_b_log = (
                        unique_models[1] if len(unique_models) > 1 else "unknown"
                    )

                    metadata = {
                        "game_name": game_name,
                        "model_a": model_a_log,
                        "model_b": model_b_log,
                        "pair_idx": match_str,
                        "roster_file": filename,
                    }

                    # Add Team-specific model info (e.g. Civilians_model: gpt-4o)
                    if len(teams) > 1:
                        for team_name in teams:
                            # Find model(s) for this team
                            team_models = set(
                                a["agent_model"]
                                for a in agents_conf
                                if a.get("team") == team_name
                            )
                            if len(team_models) == 1:
                                metadata[f"{team_name}_model"] = list(team_models)[0]
                            else:
                                metadata[f"{team_name}_model"] = (
                                    "mixed"  # Should not happen
                                )

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    log_filename = (
                        f"episode_{tag}_{game_name}_{match_str}_{timestamp}.json"
                    )
                    log_path = os.path.join("logs", log_filename)

                    env, agents = prepare_scenario(
                        env_model_name="gpt-4o",
                        agent_model_name=agent_model_list,
                        config=episode_config,
                    )

                    os.makedirs("logs", exist_ok=True)

                    await arun_one_episode(
                        env=env,
                        agent_list=agents,
                        tag=tag,
                        push_to_db=push_to_db,
                        output_path=log_path,
                        metadata=metadata,
                    )
                except Exception:
                    # TQDM will swallow prints usually, so we might want to log errors manually if crucial
                    # logging.error(f"Error in {filename}: {e}")
                    pass

        # Create tasks
        tasks = [asyncio.create_task(_worker(p)) for p in rosters_to_run]

        # Use TQDM with asyncio
        print(f"  Queuing {len(tasks)} tasks...")
        for f in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc=f"Playing {game_name}"
        ):
            await f

    print("\nAll Scheduled Rosters Executed.")


if __name__ == "__main__":
    import argparse

    # Reconfigure logging to suppress sotopia's verbose output
    # 1. Root Logger
    logging.basicConfig(level=logging.ERROR)

    # 2. Silence sotopia experimental server and generation
    server_logger = logging.getLogger("sotopia.experimental.server")
    server_logger.setLevel(logging.ERROR)
    generation_logger = logging.getLogger("sotopia.generation")
    generation_logger.setLevel(logging.ERROR)

    os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")

    parser = argparse.ArgumentParser(description="Run ELO Tournament Execution Phase")
    parser.add_argument(
        "--game",
        nargs="+",
        default=[
            "werewolves",
            "spyfall",
            "prisoners_dilemma",
            "rock_paper_scissors",
            "undercover",
        ],
        help="List of games to execute rosters for",
    )
    parser.add_argument("--tag", type=str, default="elo_exp_v1", help="Experiment tag")
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Max concurrent episodes"
    )

    args = parser.parse_args()

    asyncio.run(
        run_elo_tournament(
            game_names=args.game, tag=args.tag, concurrency_limit=args.concurrency
        )
    )
