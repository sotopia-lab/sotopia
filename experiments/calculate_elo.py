import json
import glob
import os
from collections import defaultdict
from typing import Any, Tuple, Optional

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
                    <th>ELO-Alt (Wolf/Spy)</th>
                    <th>ELO-Main (Vil/Civ)</th>
                    <th>Win Rate</th>
                    <th style="text-align: right;">Matches</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        <div class="footer">
            ELO-Alt: ELO as Werewolf (Werewolves) or Spy (Spyfall).<br>
            ELO-Main: ELO as Villager (Werewolves) or Non-Spy (Spyfall).<br>
            Symmetric games (RPS, PD) only affect Overall ELO.
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


def _get_result_werewolf(
    model_mapping: dict[str, str], agent_rewards: dict[str, float]
) -> Optional[Tuple[str, str, bool]]:
    """Return (WolfModel, VillagerModel, WolfWon)"""
    # Wolves = group of 2 agents with same reward.
    # Villagers = group of 4 agents with same reward.
    if len(model_mapping) != 6:
        return None

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
            villager_agents.extend(agents)

    if not found_wolves or len(wolf_agents) != 2 or len(villager_agents) != 4:
        return None

    wolf_reward = agent_rewards[wolf_agents[0]]
    villager_reward = agent_rewards[villager_agents[0]]

    wolf_won = wolf_reward > villager_reward

    return model_mapping[wolf_agents[0]], model_mapping[villager_agents[0]], wolf_won


def _get_result_spyfall(
    model_mapping: dict[str, str], agent_rewards: dict[str, float]
) -> Optional[Tuple[str, str, bool]]:
    """Return (SpyModel, NonSpyModel, SpyWon)"""
    # Spy = 1 agent, Non-Spy = 3 agents
    if len(model_mapping) != 4:
        # Assuming 4 player Spyfall based on standard/roster
        return None

    # Finding the unique reward isn't always reliable if everyone loses or wins,
    # but typically Spy logic: Spy wins -> High reward, Civs -> Low, or vice versa.
    # Actually, simpler: we need to know WHO is who.
    # Log doesn't explicitly store roles in parsed form easily without parsing messages or 'role' fields if available.
    # But we can infer from uniqueness if rewards differ.

    # Heuristic: Spy is the minority (size 1).
    reward_groups = defaultdict(list)
    for agent, r in agent_rewards.items():
        reward_groups[r].append(agent)

    spy_agent = None
    non_spy_agents = []

    # This relies on Spy getting a different reward OR being distinguishable.
    # If rewards are identical (e.g. Draw), we can't tell easily without roles.
    # For now, let's assume strict reward difference or we skip.

    for r, agents in reward_groups.items():
        if len(agents) == 1:
            spy_agent = agents[0]
        else:
            non_spy_agents.extend(agents)

    if not spy_agent or len(non_spy_agents) != 3:
        return None

    spy_reward = agent_rewards[spy_agent]
    non_spy_reward = agent_rewards[non_spy_agents[0]]

    spy_won = spy_reward > non_spy_reward

    return model_mapping[spy_agent], model_mapping[non_spy_agents[0]], spy_won


def _get_result_symmetric(
    model_mapping: dict[str, str], agent_rewards: dict[str, float]
) -> Optional[Tuple[str, str, float, float]]:
    """Return (ModelA, ModelB, ScoreA, ScoreB) for symmetric games."""
    agents = list(model_mapping.keys())
    # Should be 2 players usually
    if len(agents) < 2:
        return None

    # Just take first two
    a1, a2 = agents[0], agents[1]
    m1, m2 = model_mapping[a1], model_mapping[a2]
    r1, r2 = agent_rewards[a1], agent_rewards[a2]

    # Normalize scores? ELO usually expects 0-1.
    # RPS/PD might have arbitrary points (e.g. 3, 5, 0).
    # Simple binary win/loss/draw check
    if r1 > r2:
        s1, s2 = 1.0, 0.0
    elif r2 > r1:
        s1, s2 = 0.0, 1.0
    else:
        s1, s2 = 0.5, 0.5

    return m1, m2, s1, s2


def calculate_elo(log_dir: str = "logs") -> None:
    print(f"Calculating ELO from logs in: {log_dir}")

    # 1. Gather all logs
    log_files = glob.glob(os.path.join(log_dir, "*.json"))
    print(f"Found {len(log_files)} log files.")

    # Ratings
    elo_overall: dict[str, float] = defaultdict(lambda: STARTING_ELO)
    elo_wolf: dict[str, float] = defaultdict(lambda: STARTING_ELO)  # Also used for Spy
    elo_villager: dict[str, float] = defaultdict(
        lambda: STARTING_ELO
    )  # Also used for Non-Spy

    wins: dict[str, int] = defaultdict(int)
    total_games: dict[str, int] = defaultdict(int)

    processed_count = 0

    for filepath in log_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # 2. Extract Data
            model_mapping = data.get("model_mapping", {})
            rewards = data.get(
                "rewards", []
            )  # This might be list of floats or list of tuples
            environment = data.get("environment", "")
            metadata = data.get("metadata", {})
            game_name = metadata.get("game_name", "")

            if not model_mapping or not rewards:
                continue

            # Parse rewards: flatten if tuples (reward, info) -> reward
            parsed_rewards = []
            for r in rewards:
                if isinstance(r, list) or isinstance(r, tuple):
                    parsed_rewards.append(float(r[0]))
                else:
                    parsed_rewards.append(float(r))

            if len(model_mapping) != len(parsed_rewards):
                continue

            agent_rewards = {}
            for i, agent_name in enumerate(model_mapping.keys()):
                agent_rewards[agent_name] = parsed_rewards[i]

            # 3. Dispatch Logic

            # Cases: Werewolf, Spyfall (Asymmetric) vs RPS, PD (Symmetric)
            if (
                "werewolves" in game_name.lower()
                or "Werewolf" in environment
                or len(model_mapping) == 6
            ):
                # Asymmetric: Wolf vs Villager
                res = _get_result_werewolf(model_mapping, agent_rewards)
                if not res:
                    continue

                m_alt, m_main, alt_won = res

                # Update Overall
                score_alt = 1.0 if alt_won else 0.0
                score_main = 1.0 - score_alt

                # Overall Update
                elo_overall[m_alt] += K_FACTOR * (
                    score_alt - expected_score(elo_overall[m_alt], elo_overall[m_main])
                )
                elo_overall[m_main] += K_FACTOR * (
                    score_main - expected_score(elo_overall[m_main], elo_overall[m_alt])
                )

                # Split Update (Wolf vs Villager)
                elo_wolf[m_alt] += K_FACTOR * (
                    score_alt - expected_score(elo_wolf[m_alt], elo_villager[m_main])
                )
                elo_villager[m_main] += K_FACTOR * (
                    score_main - expected_score(elo_villager[m_main], elo_wolf[m_alt])
                )

                # Stats
                winner = m_alt if alt_won else m_main
                wins[winner] += 1
                total_games[m_alt] += 1
                if m_alt != m_main:
                    total_games[m_main] += 1

            elif "spyfall" in game_name.lower() or "Spyfall" in environment:
                # Asymmetric: Spy vs Non-Spy
                res = _get_result_spyfall(model_mapping, agent_rewards)
                if not res:
                    continue

                m_alt, m_main, alt_won = res  # Spy, Non-Spy

                # Update Overall
                score_alt = 1.0 if alt_won else 0.0
                score_main = 1.0 - score_alt

                elo_overall[m_alt] += K_FACTOR * (
                    score_alt - expected_score(elo_overall[m_alt], elo_overall[m_main])
                )
                elo_overall[m_main] += K_FACTOR * (
                    score_main - expected_score(elo_overall[m_main], elo_overall[m_alt])
                )

                # Split Update (Spy -> Wolf bin, Non-Spy -> Villager bin)
                elo_wolf[m_alt] += K_FACTOR * (
                    score_alt - expected_score(elo_wolf[m_alt], elo_villager[m_main])
                )
                elo_villager[m_main] += K_FACTOR * (
                    score_main - expected_score(elo_villager[m_main], elo_wolf[m_alt])
                )

                winner = m_alt if alt_won else m_main
                wins[winner] += 1
                total_games[m_alt] += 1
                if m_alt != m_main:
                    total_games[m_main] += 1

            else:
                # Symmetric (RPS, Prisoners Dilemma)
                res_sym = _get_result_symmetric(model_mapping, agent_rewards)
                if not res_sym:
                    continue

                m1, m2, s1, s2 = res_sym

                # Update Overall Only
                elo_overall[m1] += K_FACTOR * (
                    s1 - expected_score(elo_overall[m1], elo_overall[m2])
                )
                elo_overall[m2] += K_FACTOR * (
                    s2 - expected_score(elo_overall[m2], elo_overall[m1])
                )

                # For stats, assume s1 > 0.5 is win? Or detect draw?
                if s1 > s2:
                    wins[m1] += 1
                elif s2 > s1:
                    wins[m2] += 1
                # Draws don't inc wins

                total_games[m1] += 1
                if m1 != m2:
                    total_games[m2] += 1

            processed_count += 1

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
        f"{'Rank':<5} {'Model':<40} {'ELO':<8} {'ELO-Alt':<8} {'ELO-Main':<8} {'Win Rate':<10} {'Matches'}"
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
