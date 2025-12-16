import random
import os
import sys
import itertools
import json
import argparse
import glob

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.utils import load_roster_template

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


def generate_rosters(
    game_names: list[str],
    models: list[str],
    n_episodes: int = 6,
    overwrite: bool = False,
    challenger: str | None = None,
) -> None:
    """
    Generate randomized roster files for ELO tournament.
    """
    if not isinstance(game_names, list):
        game_names = [game_names]

    print(f"Generating rosters for games: {game_names}")
    print(f"Competitors: {models}")
    print(f"Episodes per matchup: {n_episodes}")
    print("\n" + "=" * 50)

    for game_name in game_names:
        print(f"Processing game: {game_name}")

        # Verify template exists
        try:
            load_roster_template(game_name)
        except Exception as e:
            print(f"  Error loading template for '{game_name}': {e}")
            continue

        if len(models) < 2:
            print(f"  Skipping {game_name}: Need at least 2 models.")
            continue

        # Output directory
        roster_output_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "rosters", game_name)
        )
        os.makedirs(roster_output_dir, exist_ok=True)
        print(f"  Output directory: {roster_output_dir}")

        # Removed directory-level safety check to allow incremental addition

        count = 0
        for pair_idx, (model_a, model_b) in enumerate(
            itertools.permutations(models, 2)
        ):
            # Challenger Mode Filter
            if challenger:
                if challenger not in (model_a, model_b):
                    continue
            for i in range(n_episodes):
                # ... (Logic setup)
                m1 = model_a
                m2 = model_b

                # Parse model name: take what's before '@', then take what's after the last '/'
                sanitized_m1 = model_a.split("@")[0].split("/")[-1]
                sanitized_m2 = model_b.split("@")[0].split("/")[-1]

                filename = f"roster_{game_name}_match{pair_idx}_ep{i}_{sanitized_m1}_vs_{sanitized_m2}.json"
                file_path = os.path.join(roster_output_dir, filename)

                # Skip if exists and not overwrite
                # IMPROVED: Check semantically (ignoring match index) to prevent duplicates if indices shift
                # New filename pattern suffix: ep{i}_{m1}_vs_{m2}.json
                semantic_suffix = f"ep{i}_{sanitized_m1}_vs_{sanitized_m2}.json"
                existing_match = glob.glob(
                    os.path.join(roster_output_dir, f"*{semantic_suffix}")
                )

                if existing_match and not overwrite:
                    # found existing file for this pair+episode
                    continue

                # Double check specific path (though glob should cover it)
                if os.path.exists(file_path) and not overwrite:
                    continue

                # 2. Assign Models
                try:
                    current_config = load_roster_template(game_name)
                except Exception as e:
                    print(f"Error loading template for '{game_name}': {e}")
                    continue

                agents = current_config["agents"]

                # Check teams
                teams = {a.get("team") for a in agents if a.get("team")}
                unique_teams = sorted([t for t in teams if t])

                if len(unique_teams) == 2:
                    # Asymmetric
                    t1, t2 = unique_teams[0], unique_teams[1]
                    for agent in agents:
                        if agent.get("team") == t1:
                            agent["agent_model"] = m1
                        elif agent.get("team") == t2:
                            agent["agent_model"] = m2
                        else:
                            agent["agent_model"] = m1
                else:
                    # Symmetric or single-team
                    for idx, agent in enumerate(agents):
                        agent["agent_model"] = m1 if idx % 2 == 0 else m2

                # 3. Randomization / Permutation
                if len(agents) == 2:
                    # Deterministic toggle for 2-player games to strictly balance speaking order
                    # i=0: [A, B], i=1: [B, A]
                    if i % 2 != 0:
                        agents.reverse()
                else:
                    # Random shuffle for multi-player games (>2 agents)
                    random.shuffle(agents)
                # Assign Names
                name_pool = NAME_POOL.copy()
                random.shuffle(name_pool)
                for idx, agent in enumerate(agents):
                    agent["name"] = name_pool[idx]

                # 4. Save
                with open(file_path, "w") as f:
                    json.dump(current_config, f, indent=4)

                count += 1

        print(f"  Generated {count} NEW rosters for {game_name}")

    print("\nGeneration Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Rosters for ELO Tournament")
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
        help="List of games",
    )
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
            "custom/Qwen/Qwen3-8B@http://127.0.0.1:1235/v1",
        ],
        help="List of models to compete",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=6,
        help="Number of episodes per matchup (default: 6)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow generation even if output directory is not empty",
    )
    parser.add_argument(
        "--challenger",
        type=str,
        default=None,
        help="If set, only generate rosters involving this model",
    )

    args = parser.parse_args()

    generate_rosters(
        game_names=args.game,
        models=args.models,
        n_episodes=args.episodes,
        overwrite=args.overwrite,
        challenger=args.challenger,
    )
