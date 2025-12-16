import json
import glob
import os
import csv
from collections import defaultdict
from typing import Any, Tuple, Optional, Dict

# Simple ELO implementation
K_FACTOR = 32
STARTING_ELO = 1200


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def generate_single_table_html(
    title: str, stats: list[dict[str, Any]], show_split_elo: bool = True
) -> str:
    """Generates the HTML for a single leaderboard table."""
    rows_html = ""
    for item in stats:
        rank = item["rank"]
        rank_display = f"#{rank}"
        if rank == 1:
            rank_display = "🥇"
        if rank == 2:
            rank_display = "🥈"
        if rank == 3:
            rank_display = "🥉"

        name = item["model"]
        provider = "Unknown"
        # Heuristic for provider
        lower_name = name.lower()
        if "gpt" in lower_name:
            provider = "OpenAI"
        elif "qwen" in lower_name:
            provider = "Alibaba"
        elif "gemini" in lower_name or "google" in lower_name:
            provider = "Google"
        elif "claude" in lower_name:
            provider = "Anthropic"
        elif "llama" in lower_name:
            provider = "Meta"
        elif "mistral" in lower_name:
            provider = "Mistral"

        wr_val = item["win_rate"]
        wr_color = "#e55" if wr_val < 50 else "#2a9d8f"

        split_elo_cells = ""
        if show_split_elo:
            split_elo_cells = f"""
            <td class="elo-split">{int(item['elo_w'])}</td>
            <td class="elo-split">{int(item['elo_v'])}</td>
            """
        else:
            split_elo_cells = """
            <td class="elo-split" style="color: #ccc;">-</td>
            <td class="elo-split" style="color: #ccc;">-</td>
            """

        row = f"""
        <tr>
            <td class="rank">{rank_display}</td>
            <td>
                <div class="model-cell">
                    <span class="model-name">{name}</span>
                    <span class="model-provider"><span class="provider-icon"></span> {provider}</span>
                </div>
            </td>
            <td class="elo">{int(item['elo'])}</td>
            {split_elo_cells}
            <td class="win-rate" style="color: {wr_color}">{item['win_rate']:.1f}%</td>
            <td class="matches">{item['matches']}</td>
        </tr>
        """
        rows_html += row

    split_headers = ""
    if show_split_elo:
        split_headers = """
                    <th>ELO-Alt (Wolf/Spy)</th>
                    <th>ELO-Main (Vil/Civ)</th>
        """
    else:
        split_headers = """
                    <th></th>
                    <th></th>
        """

    table_html = f"""
    <div class="leaderboard-section">
        <h2>{title}</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>ELO</th>
                    {split_headers}
                    <th>Win Rate</th>
                    <th style="text-align: right;">Matches</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    return table_html


def generate_html_report(tables_data: Dict[str, list[dict[str, Any]]]) -> str:
    """
    Generates the full HTML report with multiple tables.
    tables_data: { "Title": stats_list, ... }
    """

    # Generate HTML for all tables
    all_tables_html = ""

    # Ensure "Overall" comes first if present
    if "Overall" in tables_data:
        all_tables_html += generate_single_table_html(
            "Overall Leaderboard", tables_data["Overall"], show_split_elo=True
        )

    sorted_titles = sorted([t for t in tables_data.keys() if t != "Overall"])
    for title in sorted_titles:
        # Determine if we should show split ELO
        # Symmetric games: RPS, Prisoners Dilemma -> No split
        lower_title = title.lower()
        is_symmetric = "rock" in lower_title or "prisoner" in lower_title
        show_split = not is_symmetric

        all_tables_html += generate_single_table_html(
            f"{title} Leaderboard", tables_data[title], show_split_elo=show_split
        )

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Elo Leaderboard</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #ffffff; color: #333; margin: 0; padding: 40px; }}
            h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 30px; display: flex; align-items: center; gap: 10px; border-bottom: 2px solid #eee; padding-bottom: 20px; }}
            h1::before {{ content: "🏆"; font-size: 32px; }}
            h2 {{ font-size: 20px; font-weight: 600; margin-top: 40px; margin-bottom: 15px; color: #444; }}
            table {{ width: 100%; border-collapse: collapse; min-width: 800px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #f0f0f0; border-radius: 8px; overflow: hidden; }}
            th {{ text-align: left; font-size: 12px; font-weight: 700; text-transform: uppercase; color: #666; padding: 12px 16px; background-color: #f9f9f9; border-bottom: 1px solid #eee; }}
            td {{ padding: 12px 16px; border-bottom: 1px solid #f5f5f5; vertical-align: middle; }}
            tr:last-child td {{ border-bottom: none; }}
            .rank {{ width: 60px; font-weight: 700; color: #555; font-size: 16px; }}
            .model-cell {{ display: flex; flex-direction: column; }}
            .model-name {{ font-weight: 700; font-size: 15px; color: #000; }}
            .model-provider {{ font-size: 11px; color: #888; display: flex; align-items: center; gap: 4px; margin-top: 2px; }}
            .elo {{ font-weight: 700; font-size: 15px; width: 80px; }}
            .elo-split {{ font-weight: 500; font-size: 14px; width: 100px; color: #666; }}
            .win-rate {{ font-weight: 700; font-size: 15px; width: 100px; }}
            .matches {{ font-weight: 500; font-size: 14px; width: 80px; text-align: right; color: #666; }}
            .footer {{ margin-top: 50px; font-size: 13px; color: #888; border-top: 1px solid #eee; padding-top: 20px; }}
            .provider-icon {{ width: 10px; height: 10px; border-radius: 50%; background-color: #ddd; display: inline-block; }}

            /* Rank Colors */
            tr:nth-child(1) .rank {{ color: #d4af37; }}
            tr:nth-child(2) .rank {{ color: #c0c0c0; }}
            tr:nth-child(3) .rank {{ color: #cd7f32; }}
        </style>
    </head>
    <body>
        <h1>Social Games Tournament Results</h1>

        {all_tables_html}

        <div class="footer">
            <p><strong>Metrics Explanation:</strong></p>
            <ul>
                <li><strong>ELO:</strong> Overall rating across all processed games.</li>
                <li><strong>ELO-Alt:</strong> Rating as the minority/hidden role (Werewolf, Spy, Undercover).</li>
                <li><strong>ELO-Main:</strong> Rating as the majority role (Villager, Non-Spy, Civilian).</li>
                <li>Symmetric games (RPS, Prisoner's Dilemma) contribute to Overall ELO but treat roles symmetrically.</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return html_template


def get_match_result(
    model_mapping: dict[str, str],
    agent_rewards: dict[str, float],
    alt_model: str,
    main_model: str,
) -> Optional[Tuple[bool, float, float]]:
    """
    Determines if the Alt model won against the Main model.
    Returns (alt_won: bool, alt_reward, main_reward) or None if inconclusive.
    """

    # We need to find ONE representative agent for Alt model and ONE for Main model
    # to compare their rewards.
    # Why? Because in these games, team members usually get the same reward.

    alt_agent = None
    main_agent = None

    for agent, model in model_mapping.items():
        if model == alt_model and alt_agent is None:
            alt_agent = agent
        elif model == main_model and main_agent is None:
            main_agent = agent

        if alt_agent and main_agent:
            break

    if not alt_agent or not main_agent:
        return None

    r_alt = agent_rewards.get(alt_agent, 0.0)
    r_main = agent_rewards.get(main_agent, 0.0)

    return (r_alt > r_main), r_alt, r_main


def process_logs(log_files: list[str]) -> list[dict[str, Any]]:
    """
    Process a list of log files and return stats for the leaderboard.
    """
    elo_overall: dict[str, float] = defaultdict(lambda: STARTING_ELO)
    elo_wolf: dict[str, float] = defaultdict(lambda: STARTING_ELO)  # Alt role
    elo_villager: dict[str, float] = defaultdict(lambda: STARTING_ELO)  # Main role

    wins: dict[str, int] = defaultdict(int)
    total_games: dict[str, int] = defaultdict(int)

    count = 0
    for filepath in log_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            model_mapping = data.get("model_mapping", {})
            rewards = data.get("rewards", [])
            metadata = data.get("metadata", {})

            if not model_mapping or not rewards:
                continue

            parsed_rewards = []
            for r in rewards:
                if isinstance(r, (list, tuple)):
                    parsed_rewards.append(float(r[0]))
                else:
                    parsed_rewards.append(float(r))

            if len(model_mapping) != len(parsed_rewards):
                continue

            # Map Agent Name -> Reward
            # Relies on implicit ordering of keys vs list.
            # Sotopia seems to maintain this consistency.
            agent_rewards = {}
            for i, agent_name in enumerate(model_mapping.keys()):
                agent_rewards[agent_name] = parsed_rewards[i]

            # --- Dispatch Logic based on Metadata Keys ---

            check_processed = False

            # 1. Werewolves
            if "Werewolves_model" in metadata and "Villagers_model" in metadata:
                m_alt = metadata["Werewolves_model"]
                m_main = metadata["Villagers_model"]

                res = get_match_result(model_mapping, agent_rewards, m_alt, m_main)
                if res:
                    alt_won, r_alt, r_main = res
                    score_alt = 1.0 if alt_won else 0.0
                    score_main = 1.0 - score_alt

                    # Updates
                    elo_overall[m_alt] += K_FACTOR * (
                        score_alt
                        - expected_score(elo_overall[m_alt], elo_overall[m_main])
                    )
                    elo_overall[m_main] += K_FACTOR * (
                        score_main
                        - expected_score(elo_overall[m_main], elo_overall[m_alt])
                    )

                    elo_wolf[m_alt] += K_FACTOR * (
                        score_alt
                        - expected_score(elo_wolf[m_alt], elo_villager[m_main])
                    )
                    elo_villager[m_main] += K_FACTOR * (
                        score_main
                        - expected_score(elo_villager[m_main], elo_wolf[m_alt])
                    )

                    winner = m_alt if alt_won else m_main
                    wins[winner] += 1
                    total_games[m_alt] += 1
                    if m_alt != m_main:
                        total_games[m_main] += 1
                    check_processed = True

            # 2. Spyfall
            elif "Spy_model" in metadata and "Non-Spies_model" in metadata:
                m_alt = metadata["Spy_model"]
                m_main = metadata["Non-Spies_model"]

                res = get_match_result(model_mapping, agent_rewards, m_alt, m_main)
                if res:
                    alt_won, r_alt, r_main = res
                    score_alt = 1.0 if alt_won else 0.0
                    score_main = 1.0 - score_alt

                    elo_overall[m_alt] += K_FACTOR * (
                        score_alt
                        - expected_score(elo_overall[m_alt], elo_overall[m_main])
                    )
                    elo_overall[m_main] += K_FACTOR * (
                        score_main
                        - expected_score(elo_overall[m_main], elo_overall[m_alt])
                    )

                    elo_wolf[m_alt] += K_FACTOR * (
                        score_alt
                        - expected_score(elo_wolf[m_alt], elo_villager[m_main])
                    )
                    elo_villager[m_main] += K_FACTOR * (
                        score_main
                        - expected_score(elo_villager[m_main], elo_wolf[m_alt])
                    )

                    winner = m_alt if alt_won else m_main
                    wins[winner] += 1
                    total_games[m_alt] += 1
                    if m_alt != m_main:
                        total_games[m_main] += 1
                    check_processed = True

            # 3. Undercover
            elif "Undercover_model" in metadata and "Civilians_model" in metadata:
                m_alt = metadata["Undercover_model"]
                m_main = metadata["Civilians_model"]

                res = get_match_result(model_mapping, agent_rewards, m_alt, m_main)
                if res:
                    alt_won, r_alt, r_main = res
                    score_alt = 1.0 if alt_won else 0.0
                    score_main = 1.0 - score_alt

                    elo_overall[m_alt] += K_FACTOR * (
                        score_alt
                        - expected_score(elo_overall[m_alt], elo_overall[m_main])
                    )
                    elo_overall[m_main] += K_FACTOR * (
                        score_main
                        - expected_score(elo_overall[m_main], elo_overall[m_alt])
                    )

                    elo_wolf[m_alt] += K_FACTOR * (
                        score_alt
                        - expected_score(elo_wolf[m_alt], elo_villager[m_main])
                    )
                    elo_villager[m_main] += K_FACTOR * (
                        score_main
                        - expected_score(elo_villager[m_main], elo_wolf[m_alt])
                    )

                    winner = m_alt if alt_won else m_main
                    wins[winner] += 1
                    total_games[m_alt] += 1
                    if m_alt != m_main:
                        total_games[m_main] += 1
                    check_processed = True

            # 4. Symmetric Fallback
            else:
                # If no role keys, assume symmetric (RPS, PD, etc)
                # metadata should have model_a, model_b
                # OR we just pick 2 from model_mapping

                agents = list(model_mapping.keys())
                if len(agents) >= 2:
                    # Prefer metadata definition if available?
                    # Actually standardizing on model_mapping logic is safer

                    a1, a2 = agents[0], agents[1]
                    m1, m2 = model_mapping[a1], model_mapping[a2]
                    r1, r2 = agent_rewards[a1], agent_rewards[a2]

                    if r1 > r2:
                        s1, s2 = 1.0, 0.0
                    elif r2 > r1:
                        s1, s2 = 0.0, 1.0
                    else:
                        s1, s2 = 0.5, 0.5

                    elo_overall[m1] += K_FACTOR * (
                        s1 - expected_score(elo_overall[m1], elo_overall[m2])
                    )
                    elo_overall[m2] += K_FACTOR * (
                        s2 - expected_score(elo_overall[m2], elo_overall[m1])
                    )

                    if s1 > s2:
                        wins[m1] += 1
                    elif s2 > s1:
                        wins[m2] += 1

                    total_games[m1] += 1
                    if m1 != m2:
                        total_games[m2] += 1
                    check_processed = True

            if check_processed:
                count += 1

        except Exception:
            # print(f"Error processing {filepath}: {e}")
            continue

    # Generate Stats List
    sorted_models = sorted(
        elo_overall.keys(), key=lambda m: elo_overall[m], reverse=True
    )
    stats_list = []

    for rank, model in enumerate(sorted_models, 1):
        n_games = total_games[model]
        win_rate = (wins[model] / n_games * 100) if n_games > 0 else 0.0
        display_name = model.split("@")[0].replace("custom/", "")

        stats_list.append(
            {
                "rank": rank,
                "model": display_name,
                "elo": elo_overall[model],
                "elo_w": elo_wolf[model],
                "elo_v": elo_villager[model],
                "win_rate": win_rate,
                "matches": n_games,
            }
        )

    return stats_list


def save_to_csv(title: str, stats: list[dict[str, Any]]) -> None:
    """Saves the stats list to a CSV file."""
    # Sanitize title to filename
    safe_title = title.lower().replace(" ", "_").replace("/", "_")
    filename = os.path.join("experiments", f"elo_results_{safe_title}.csv")

    headers = [
        "Rank",
        "Model",
        "ELO",
        "ELO-Alt (Wolf/Spy)",
        "ELO-Main (Vil/Civ)",
        "Win Rate",
        "Matches",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for item in stats:
            writer.writerow(
                [
                    item["rank"],
                    item["model"],
                    int(item["elo"]),
                    int(item["elo_w"]),
                    int(item["elo_v"]),
                    f"{item['win_rate']:.1f}%",
                    item["matches"],
                ]
            )
    print(f"Generated CSV: {filename}")


def calculate_elo(log_dir: str = "logs") -> None:
    print(f"Calculating ELO from logs in: {log_dir}")

    # Gather logs
    log_files = glob.glob(os.path.join(log_dir, "*.json"))

    # User requested to process ALL logs, no filtering.
    filtered_Logs = log_files

    print(f"Found {len(filtered_Logs)} items")

    # 1. Group logs by Game
    logs_by_game: Dict[str, list[str]] = defaultdict(list)

    for filepath in filtered_Logs:
        try:
            with open(filepath, "r") as f:
                header = json.load(f)
                metadata = header.get("metadata", {})
                game_name = metadata.get("game_name", "Unknown")

                # Robust detection if metadata missing but keys present
                if game_name == "Unknown":
                    # Fallback check based on keys
                    if "Werewolves_model" in metadata:
                        game_name = "Werewolves"
                    elif "Spy_model" in metadata:
                        game_name = "Spyfall"
                    elif "Undercover_model" in metadata:
                        game_name = "Undercover"
                    else:
                        # Fallback to env string
                        print("Unknown game name, falling back to env string")
                        env = header.get("environment", "")
                        if "werewolf" in env.lower():
                            game_name = "werewolves"
                        elif "spyfall" in env.lower():
                            game_name = "spyfall"
                        elif "prison" in env.lower():
                            game_name = "prisoners_dilemma"
                        elif "rock" in env.lower():
                            game_name = "rock_paper_scissors"

                logs_by_game[game_name].append(filepath)
        except Exception:
            continue

    # 2. Calculate Stats
    all_tables_data = {}

    # Overall
    print("Processing Overall...")
    overall_stats = process_logs(filtered_Logs)
    all_tables_data["Overall"] = overall_stats
    save_to_csv("Overall", overall_stats)

    # Per Game
    for game_name, game_logs in logs_by_game.items():
        if not game_name or game_name == "Unknown":
            continue
        print(f"Processing {game_name} ({len(game_logs)} games)...")
        # Format title case
        title = game_name.replace("_", " ").title()
        stats = process_logs(game_logs)
        all_tables_data[title] = stats
        save_to_csv(title, stats)

    # 3. Generate HTML
    html_content = generate_html_report(all_tables_data)
    output_html = os.path.join("experiments", "elo_leaderboard.html")
    with open(output_html, "w") as f:
        f.write(html_content)

    print(f"\nSuccessfully generated {output_html}")
    print("Tables generated for:", ", ".join(all_tables_data.keys()))


if __name__ == "__main__":
    calculate_elo()
