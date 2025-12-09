import json
import glob
import os
from collections import defaultdict
from typing import Any

# Simple ELO implementation
K_FACTOR = 32
STARTING_ELO = 1200


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def generate_html_report(stats: list[dict[str, Any]]) -> str:
    """
    Generates an HTML file mimicking the sleek leaderboard design.
    stats: list of dicts with keys: rank, model, elo, elo_w, elo_v, win_rate, matches
    """

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Elo Leaderboard</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #ffffff; color: #333; margin: 0; padding: 40px; }
            h1 { font-size: 24px; font-weight: 700; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
            h1::before { content: "🏆"; font-size: 28px; }
            table { width: 100%; border-collapse: collapse; min-width: 800px; }
            th { text-align: left; font-size: 12px; font-weight: 700; text-transform: uppercase; color: #888; padding: 12px 0; border-bottom: 1px solid #eee; }
            td { padding: 16px 0; border-bottom: 1px solid #f5f5f5; vertical-align: middle; }
            .rank { width: 60px; font-weight: 700; color: #555; font-size: 16px; }
            .medal { font-size: 20px; margin-right: 5px; }
            .model-cell { display: flex; flex-direction: column; }
            .model-name { font-weight: 700; font-size: 16px; color: #000; }
            .model-provider { font-size: 12px; color: #888; display: flex; align-items: center; gap: 4px; margin-top: 4px; }
            .elo { font-weight: 700; font-size: 16px; width: 100px; }
            .elo-split { font-weight: 700; font-size: 16px; width: 100px; }
            .win-rate { font-weight: 700; font-size: 16px; width: 120px; color: #e55; }
            .matches { font-weight: 500; font-size: 16px; width: 80px; text-align: right; color: #333; }
            .footer { margin-top: 30px; font-size: 14px; color: #666; font-weight: 500; }
            .provider-icon { width: 14px; height: 14px; border-radius: 50%; background-color: #ddd; display: inline-block; }

            /* Rank Colors */
            tr:nth-child(1) .rank { color: #d4af37; } /* Gold-ish logic handled by emoji */
            tr:nth-child(2) .rank { color: #c0c0c0; }
            tr:nth-child(3) .rank { color: #cd7f32; }
        </style>
    </head>
    <body>
        <h1>Elo Leaderboard</h1>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>ELO</th>
                    <th>ELO-W</th>
                    <th>ELO-V</th>
                    <th>Win Rate</th>
                    <th style="text-align: right;">Matches</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        <div class="footer">
            ELO-W = Elo as wolf; ELO-V = Elo as villager.
        </div>
    </body>
    </html>
    """

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

        # Infer provider/company
        name = item["model"]
        provider = "Unknown"
        if "gpt" in name:
            provider = "OpenAI"
        elif "qwen" in name:
            provider = "Alibaba"
        elif "gemini" in name or "google" in name:
            provider = "Google"
        elif "claude" in name:
            provider = "Anthropic"
        elif "llama" in name:
            provider = "Meta"

        # Color scale for win rate?
        wr_val = item["win_rate"]
        wr_color = "#e55" if wr_val < 50 else "#2a9d8f"

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
            <td class="elo-split">{int(item['elo_w'])}</td>
            <td class="elo-split">{int(item['elo_v'])}</td>
            <td class="win-rate" style="color: {wr_color}">{item['win_rate']:.1f}%</td>
            <td class="matches">{item['matches']}</td>
        </tr>
        """
        rows_html += row

    return html_template.replace("{rows}", rows_html)


def calculate_elo(log_dir: str = "logs") -> None:
    print(f"Calculating ELO from logs in: {log_dir}")

    # 1. Gather all logs
    log_files = glob.glob(os.path.join(log_dir, "*.json"))
    print(f"Found {len(log_files)} log files.")

    # Ratings
    elo_overall: dict[str, float] = defaultdict(lambda: STARTING_ELO)
    elo_wolf: dict[str, float] = defaultdict(lambda: STARTING_ELO)
    elo_villager: dict[str, float] = defaultdict(lambda: STARTING_ELO)

    wins: dict[str, int] = defaultdict(int)
    total_games: dict[str, int] = defaultdict(int)

    processed_count = 0

    for filepath in log_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # 2. Extract Data
            model_mapping = data.get("model_mapping", {})
            rewards = data.get("rewards", [])

            if not model_mapping or not rewards:
                continue

            if len(model_mapping) != len(rewards):
                continue

            # 3. Identify Roles and Winner
            # Robust logic: "There are only two wolves".
            # Wolves = group of 2 agents with same reward.
            # Villagers = group of 4 agents with same reward.

            if len(model_mapping) != 6:
                # If not 6 player game, skip
                continue

            agent_rewards = {}
            for i, agent_name in enumerate(model_mapping.keys()):
                agent_rewards[agent_name] = rewards[i]

            # Group agents by reward value
            reward_groups = defaultdict(list)
            for agent, r in agent_rewards.items():
                reward_groups[r].append(agent)

            wolf_agents = []
            villager_agents = []

            found_wolves = False
            for r, agents in reward_groups.items():
                if len(agents) == 2:
                    wolf_agents = agents
                    found_wolves = True
                else:
                    villager_agents.extend(
                        agents
                    )  # Collect all others as villagers (usually 4)

            if not found_wolves or len(wolf_agents) != 2 or len(villager_agents) != 4:
                # Unexpected structure (e.g. 3v3 or equal split)
                continue

            # Determine Winner
            # Assuming strictly positive reward = win, negative = loss?
            # Or just highest reward wins?
            # In log provided: Villagers got 1.0, Wolves got -1.0. 1.0 > -1.0 -> Villagers Win.

            wolf_reward = agent_rewards[wolf_agents[0]]
            villager_reward = agent_rewards[villager_agents[0]]

            wolf_won = False
            if wolf_reward > villager_reward:
                wolf_won = True

            # Identify Models (Representative for 1v1 ELO update)
            # Tournament is 1 model type vs 1 model type
            w_model = model_mapping[wolf_agents[0]]
            v_model = model_mapping[villager_agents[0]]

            # (If mixed models on same team, this takes just one, but based on user context it's usually model A vs model B)

            processed_count += 1

            # 4. Update ELO
            # Score for Wolf Team
            score_w = 1.0 if wolf_won else 0.0
            score_v = 1.0 - score_w

            # -- Overall ELO --
            rating_w_overall = elo_overall[w_model]
            rating_v_overall = elo_overall[v_model]
            exp_w_overall = expected_score(rating_w_overall, rating_v_overall)

            elo_overall[w_model] += K_FACTOR * (score_w - exp_w_overall)
            elo_overall[v_model] += K_FACTOR * (score_v - (1.0 - exp_w_overall))

            # -- Split ELO --
            rat_w_split = elo_wolf[w_model]
            rat_v_split = elo_villager[v_model]

            exp_w_split = expected_score(rat_w_split, rat_v_split)

            elo_wolf[w_model] += K_FACTOR * (score_w - exp_w_split)
            elo_villager[v_model] += K_FACTOR * (score_v - (1.0 - exp_w_split))

            # Updates Stats
            if wolf_won:
                wins[w_model] += 1
            else:
                wins[v_model] += 1

            total_games[w_model] += 1
            if w_model != v_model:
                total_games[v_model] += 1

        except Exception:
            # print(f"Error processing {filepath}: {e}")
            continue

    print(f"Processed {processed_count} valid games.")

    # 5. Print Results
    sorted_models = sorted(
        elo_overall.keys(), key=lambda m: elo_overall[m], reverse=True
    )
    stats_list = []

    print("\n" + "=" * 115)
    print(
        f"{'Rank':<5} {'Model':<40} {'ELO':<8} {'ELO-W':<8} {'ELO-V':<8} {'Win Rate':<10} {'Matches'}"
    )
    print("-" * 115)

    for rank, model in enumerate(sorted_models, 1):
        n_games = total_games[model]
        win_rate = (wins[model] / n_games * 100) if n_games > 0 else 0.0

        display_name = model.split("@")[0].replace("custom/", "")
        if len(display_name) > 38:
            display_name = display_name[:35] + "..."

        # Raw Ratings
        print(
            f"{rank:<5} {display_name:<40} {elo_overall[model]:<8.0f} {elo_wolf[model]:<8.0f} {elo_villager[model]:<8.0f} {win_rate:<10.1f}% {n_games}"
        )

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
    print("=" * 115 + "\n")

    # Generate HTML
    html_content = generate_html_report(stats_list)
    output_html = "elo_leaderboard.html"
    with open(output_html, "w") as f:
        f.write(html_content)
    print(f"Generate HTML report: {output_html}")


if __name__ == "__main__":
    calculate_elo()
