"""Launcher for the Duskmire Werewolves social game scenario with human player."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union, cast
from datetime import datetime
import webbrowser

import redis

from sotopia.agents import LLMAgent, HumanAgent
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipType,
)
from sotopia.envs import SocialGameEnv
from sotopia.envs.evaluators import (
    EpisodeLLMEvaluator,
    EvaluationForAgents,
    RuleBasedTerminatedEvaluator,
)
from sotopia.server import arun_one_episode
from sotopia.database import SotopiaDimensions

BASE_DIR = Path(__file__).resolve().parent
ROLE_ACTIONS_PATH = BASE_DIR / "role_actions.json"
RULEBOOK_PATH = BASE_DIR / "game_rules.json"
ROSTER_PATH = BASE_DIR / "roster.json"
PLAYER_VIEW_HTML = BASE_DIR / "player_view.html"

os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")
redis.Redis(host="localhost", port=6379)

COMMON_GUIDANCE = (
    "During your turn you must respond. If 'action' is available, use commands like 'kill NAME', "
    "'inspect NAME', 'save NAME', 'poison NAME', or 'vote NAME'. Werewolf night speech is private to the pack. "
    "Day discussion is public. Voting requires an 'action' beginning with 'vote'."
)


class PlayerView:
    """Manages the HTML output for player-visible game information."""

    def __init__(
        self,
        output_path: Path,
        player_name: str,
        role: str,
        all_player_names: List[str],
    ):
        self.output_path = output_path
        self.player_name = player_name
        self.role = role
        self.all_player_names = all_player_names
        self.events: List[str] = []
        self.input_file = output_path.parent / "player_input.json"
        self.waiting_for_input = False
        self.available_actions: List[str] = []
        self._initialize_html()

    def _initialize_html(self) -> None:
        """Create the initial HTML file."""
        player_list = "\n".join([f"<li>{name}</li>" for name in self.all_player_names])

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Duskmire Werewolves - {self.player_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            gap: 20px;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
        }}
        .sidebar {{
            width: 250px;
            flex-shrink: 0;
        }}
        .main-content {{
            flex: 1;
            max-width: 900px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: white;
            font-size: 24px;
        }}
        .header .info {{
            color: #f0f0f0;
            font-size: 14px;
        }}
        .players-box {{
            background: #16213e;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .players-box h3 {{
            margin-top: 0;
            color: #667eea;
            font-size: 16px;
        }}
        .players-box ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .players-box li {{
            padding: 8px;
            margin: 5px 0;
            background: #1a1a2e;
            border-radius: 5px;
            font-size: 14px;
        }}
        .input-box {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            position: sticky;
            top: 20px;
        }}
        .input-box h3 {{
            margin-top: 0;
            color: #667eea;
            font-size: 16px;
        }}
        .input-box.waiting {{
            background: #27ae60;
            animation: pulse 2s infinite;
        }}
        .input-box.disabled {{
            opacity: 0.5;
        }}
        .action-buttons {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }}
        .action-btn {{
            flex: 1;
            padding: 10px;
            border: 2px solid #667eea;
            background: #1a1a2e;
            color: #eee;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }}
        .action-btn:hover {{
            background: #667eea;
        }}
        .action-btn.selected {{
            background: #667eea;
            border-color: #764ba2;
        }}
        .action-btn:disabled {{
            cursor: not-allowed;
            opacity: 0.3;
        }}
        textarea {{
            width: 100%;
            padding: 10px;
            background: #1a1a2e;
            border: 2px solid #667eea;
            border-radius: 5px;
            color: #eee;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
            min-height: 80px;
            box-sizing: border-box;
        }}
        .submit-btn {{
            width: 100%;
            padding: 12px;
            background: #27ae60;
            border: none;
            border-radius: 5px;
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 10px;
        }}
        .submit-btn:hover {{
            background: #2ecc71;
        }}
        .submit-btn:disabled {{
            background: #555;
            cursor: not-allowed;
        }}
        .event {{
            background: #16213e;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 5px;
        }}
        .event.phase {{
            border-left-color: #f39c12;
            background: #2c3e50;
        }}
        .event.death {{
            border-left-color: #e74c3c;
            background: #341a1a;
        }}
        .event.speak {{
            border-left-color: #3498db;
        }}
        .event.action {{
            border-left-color: #e67e22;
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 12px;
            margin-bottom: 5px;
        }}
        .speaker {{
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .content {{
            line-height: 1.6;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.8; }}
        }}
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="players-box">
            <h3>👥 Players</h3>
            <ul>
{player_list}
            </ul>
        </div>

        <div class="input-box" id="inputBox">
            <h3>🎮 Your Action</h3>
            <div id="status">Waiting for your turn...</div>
            <div id="inputControls" style="display:none;">
                <div class="action-buttons" id="actionButtons"></div>
                <textarea id="argument" placeholder="Enter your message or action..."></textarea>
                <button class="submit-btn" onclick="submitAction()">Submit Action</button>
            </div>
        </div>
    </div>

    <div class="main-content">
        <div class="header">
            <h1>🌕 Duskmire Werewolves</h1>
            <div class="info">
                <strong>You are:</strong> {self.player_name} | <strong>Role:</strong> {self.role}
            </div>
        </div>
        <div id="events">
            <div class="event phase">
                <div class="timestamp">{datetime.now().strftime('%H:%M:%S')}</div>
                <div class="content">Game starting...</div>
            </div>
        </div>
    </div>

    <script>
        let selectedAction = null;

        function selectAction(action) {{
            selectedAction = action;
            document.querySelectorAll('.action-btn').forEach(btn => {{
                btn.classList.remove('selected');
            }});
            event.target.classList.add('selected');

            // Show/hide textarea based on action type
            const textarea = document.getElementById('argument');
            if (action === 'none') {{
                textarea.style.display = 'none';
                textarea.value = '';
            }} else {{
                textarea.style.display = 'block';
                textarea.focus();
            }}
        }}

        function submitAction() {{
            if (!selectedAction) {{
                alert('Please select an action first');
                return;
            }}

            const argument = document.getElementById('argument').value;
            const data = {{
                action_type: selectedAction,
                argument: argument,
                timestamp: new Date().toISOString()
            }};

            // Write to input file
            fetch('player_input.json', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify(data)
            }}).catch(() => {{
                // Fallback: Try to save using file system (won't work in browser)
                console.log('Action:', data);
            }});

            // Disable controls
            document.getElementById('inputControls').style.display = 'none';
            document.getElementById('status').textContent = 'Action submitted! Waiting for next turn...';
            document.getElementById('inputBox').classList.remove('waiting');
        }}

        // Auto-scroll to bottom
        function scrollToBottom() {{
            window.scrollTo(0, document.body.scrollHeight);
        }}
        setInterval(scrollToBottom, 2000);
    </script>
</body>
</html>"""
        self.output_path.write_text(html)

    def add_event(self, event_type: str, content: str, speaker: str = "") -> None:
        """Add a new event to the player view."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        speaker_html = f'<div class="speaker">{speaker}</div>' if speaker else ""

        event_html = f"""
        <div class="event {event_type}">
            <div class="timestamp">{timestamp}</div>
            {speaker_html}
            <div class="content">{content}</div>
        </div>"""

        self.events.append(event_html)
        self._update_html()

    def enable_input(self, available_actions: List[str]) -> None:
        """Enable the input controls with available actions."""
        self.waiting_for_input = True
        self.available_actions = available_actions
        # Clear any previous input
        if self.input_file.exists():
            self.input_file.unlink()
        self._update_html()

    def wait_for_input(self) -> dict[str, str]:
        """Wait for player input from the HTML interface."""
        import time

        while not self.input_file.exists():
            time.sleep(0.5)

        try:
            data = json.loads(self.input_file.read_text())
            self.waiting_for_input = False
            self._update_html()
            return data
        except (json.JSONDecodeError, KeyError):
            # If file is corrupt, wait and try again
            time.sleep(0.5)
            return self.wait_for_input()

    def _update_html(self) -> None:
        """Update the HTML file with all events and dynamic input state."""
        events_html = "\n".join(self.events)
        player_list = "\n".join([f"<li>{name}</li>" for name in self.all_player_names])

        # Generate action buttons HTML based on available actions
        action_buttons_html = ""
        input_display = "none"
        status_text = "Waiting for your turn..."
        input_box_class = ""

        if self.waiting_for_input and self.available_actions:
            input_display = "block"
            status_text = "🎮 YOUR TURN! Select an action:"
            input_box_class = "waiting"
            action_buttons = []
            for action in self.available_actions:
                action_label = action.replace("_", " ").title()
                action_buttons.append(
                    f'<button class="action-btn" onclick="selectAction(\'{action}\')">{action_label}</button>'
                )
            action_buttons_html = "\n".join(action_buttons)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Duskmire Werewolves - {self.player_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            gap: 20px;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
        }}
        .sidebar {{
            width: 250px;
            flex-shrink: 0;
        }}
        .main-content {{
            flex: 1;
            max-width: 900px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: white;
            font-size: 24px;
        }}
        .header .info {{
            color: #f0f0f0;
            font-size: 14px;
        }}
        .players-box {{
            background: #16213e;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .players-box h3 {{
            margin-top: 0;
            color: #667eea;
            font-size: 16px;
        }}
        .players-box ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .players-box li {{
            padding: 8px;
            margin: 5px 0;
            background: #1a1a2e;
            border-radius: 5px;
            font-size: 14px;
        }}
        .input-box {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            position: sticky;
            top: 20px;
        }}
        .input-box h3 {{
            margin-top: 0;
            color: #667eea;
            font-size: 16px;
        }}
        .input-box.waiting {{
            background: #27ae60;
            animation: pulse 2s infinite;
        }}
        .action-buttons {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        .action-btn {{
            flex: 1;
            min-width: 80px;
            padding: 10px;
            border: 2px solid #667eea;
            background: #1a1a2e;
            color: #eee;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }}
        .action-btn:hover {{
            background: #667eea;
        }}
        .action-btn.selected {{
            background: #667eea;
            border-color: #764ba2;
        }}
        textarea {{
            width: 100%;
            padding: 10px;
            background: #1a1a2e;
            border: 2px solid #667eea;
            border-radius: 5px;
            color: #eee;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
            min-height: 80px;
            box-sizing: border-box;
        }}
        .submit-btn {{
            width: 100%;
            padding: 12px;
            background: #27ae60;
            border: none;
            border-radius: 5px;
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 10px;
        }}
        .submit-btn:hover {{
            background: #2ecc71;
        }}
        .event {{
            background: #16213e;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 5px;
        }}
        .event.phase {{
            border-left-color: #f39c12;
            background: #2c3e50;
        }}
        .event.death {{
            border-left-color: #e74c3c;
            background: #341a1a;
        }}
        .event.speak {{
            border-left-color: #3498db;
        }}
        .event.action {{
            border-left-color: #e67e22;
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 12px;
            margin-bottom: 5px;
        }}
        .speaker {{
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .content {{
            line-height: 1.6;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.8; }}
        }}
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="players-box">
            <h3>👥 Players</h3>
            <ul>
{player_list}
            </ul>
        </div>

        <div class="input-box {input_box_class}" id="inputBox">
            <h3>🎮 Your Action</h3>
            <div id="status">{status_text}</div>
            <div id="inputControls" style="display:{input_display};">
                <div class="action-buttons" id="actionButtons">
{action_buttons_html}
                </div>
                <textarea id="argument" placeholder="Enter your message or action..." style="display:none;"></textarea>
                <button class="submit-btn" onclick="submitAction()">Submit Action</button>
            </div>
        </div>
    </div>

    <div class="main-content">
        <div class="header">
            <h1>🌕 Duskmire Werewolves</h1>
            <div class="info">
                <strong>You are:</strong> {self.player_name} | <strong>Role:</strong> {self.role}
            </div>
        </div>
        <div id="events">
{events_html}
        </div>
    </div>

    <script>
        let selectedAction = null;
        let lastEventCount = 0;

        function selectAction(action) {{
            selectedAction = action;
            document.querySelectorAll('.action-btn').forEach(btn => {{
                btn.classList.remove('selected');
            }});
            event.target.classList.add('selected');

            // Show/hide textarea based on action type
            const textarea = document.getElementById('argument');
            if (action === 'none') {{
                textarea.style.display = 'none';
                textarea.value = '';
            }} else {{
                textarea.style.display = 'block';
                textarea.focus();
            }}
        }}

        function submitAction() {{
            if (!selectedAction) {{
                alert('Please select an action first');
                return;
            }}

            const argument = document.getElementById('argument').value;
            const data = {{
                action_type: selectedAction,
                argument: argument,
                timestamp: new Date().toISOString()
            }};

            // Try to write file via fetch
            fetch('save_action', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify(data)
            }}).then(() => {{
                // Disable controls
                document.getElementById('inputControls').style.display = 'none';
                document.getElementById('status').textContent = 'Action submitted! Waiting for next turn...';
                document.getElementById('inputBox').classList.remove('waiting');
                selectedAction = null;
                document.getElementById('argument').value = '';
            }}).catch((err) => {{
                console.error('Failed to save action:', err);
                alert('Failed to submit action. Please try again.');
            }});
        }}

        // Poll for updates without full page reload
        function checkForUpdates() {{
            // Only reload if we're not currently inputting
            const inputBox = document.getElementById('inputBox');
            const isWaiting = inputBox.classList.contains('waiting');

            // If input is not active, check for updates
            if (!isWaiting) {{
                fetch(window.location.href + '?t=' + Date.now())
                    .then(r => r.text())
                    .then(html => {{
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(html, 'text/html');
                        const newInputBox = doc.getElementById('inputBox');
                        const newEvents = doc.getElementById('events');

                        // Update events area
                        if (newEvents) {{
                            document.getElementById('events').innerHTML = newEvents.innerHTML;
                            scrollToBottom();
                        }}

                        // Update input box state
                        if (newInputBox && newInputBox.classList.contains('waiting')) {{
                            location.reload(); // Reload when it's our turn
                        }}
                    }})
                    .catch(() => {{}});
            }}
        }}

        // Auto-scroll to bottom
        function scrollToBottom() {{
            const mainContent = document.querySelector('.main-content');
            if (mainContent) {{
                mainContent.scrollTop = mainContent.scrollHeight;
            }}
        }}

        scrollToBottom();
        // Check for updates every 2 seconds
        setInterval(checkForUpdates, 2000);
    </script>
</body>
</html>"""
        self.output_path.write_text(html)


class PlayerViewHumanAgent(HumanAgent):
    """HumanAgent that also writes to PlayerView HTML and reads input from it."""

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: Any | None = None,
        available_agent_names: list[str] | None = None,
        player_view: PlayerView | None = None,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
            available_agent_names=available_agent_names,
        )
        self.player_view = player_view

    async def aact(self, obs: Any) -> Any:
        """Act and update player view with relevant information."""
        from sotopia.messages import AgentAction

        self.recv_message("Environment", obs)

        # Parse observation to extract player-visible information
        if self.player_view and hasattr(obs, "to_natural_language"):
            obs_text = obs.to_natural_language()

            # Parse line by line to avoid duplicates and properly categorize events
            lines = obs_text.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for game over / winner announcement
                if (
                    "GAME OVER" in line
                    or ("Werewolves win" in line)
                    or ("Villagers win" in line)
                ):
                    self.player_view.add_event("phase", f"🎮 GAME OVER: {line}")

                # Check for voting results and eliminations
                elif (
                    "voted for" in line
                    or "has been eliminated" in line
                    or "was eliminated" in line
                ):
                    self.player_view.add_event(
                        "action", line.replace("[God]", "").strip()
                    )

                # Check for death announcements
                elif "was found dead" in line or "died" in line:
                    self.player_view.add_event(
                        "death", line.replace("[God]", "").strip()
                    )

                # Check for phase announcements
                elif "Night phase begins" in line:
                    if not (
                        self.player_view.events
                        and "Night phase begins" in self.player_view.events[-1]
                    ):
                        self.player_view.add_event(
                            "phase", "🌙 Night phase begins. Stay quiet..."
                        )

                elif "Day discussion starts" in line or (
                    "Phase: 'day_discussion' begins" in line
                ):
                    if not (
                        self.player_view.events
                        and "Day breaks" in self.player_view.events[-1]
                    ):
                        self.player_view.add_event(
                            "phase", "☀️ Day breaks. Time to discuss!"
                        )

                elif "Voting phase" in line or ("Phase: 'voting' begins" in line):
                    if not (
                        self.player_view.events
                        and "Voting phase" in self.player_view.events[-1]
                    ):
                        self.player_view.add_event(
                            "phase", "🗳️ Voting phase. Time to make your choice."
                        )

                # Check for speech from players (avoid God messages and duplicates)
                elif (" said:" in line or " says:" in line) and "[God]" not in line:
                    parts = line.split(" said:" if " said:" in line else " says:")
                    if len(parts) == 2:
                        speaker = parts[0].strip()
                        message = parts[1].strip().strip('"')
                        # Check if not duplicate
                        if not (
                            self.player_view.events
                            and speaker in self.player_view.events[-1]
                            and message in self.player_view.events[-1]
                        ):
                            self.player_view.add_event("speak", message, speaker)

        # Get available actions from observation
        available_actions = (
            obs.available_actions if hasattr(obs, "available_actions") else ["none"]
        )

        if available_actions != ["none"] and self.player_view:
            # Enable HTML input and wait for player response
            self.player_view.enable_input(available_actions)
            print(
                f"\n🎮 Waiting for {self.agent_name}'s input in the HTML interface..."
            )

            # Wait for input from HTML
            input_data = self.player_view.wait_for_input()
            action_type = input_data.get("action_type", "none")
            argument = input_data.get("argument", "")

            # Enhanced voting support
            if action_type == "action" and argument.lower().startswith("vote"):
                name_part = argument[4:].strip()
                if name_part and self.available_agent_names:
                    matched_name = self._find_matching_name(name_part)
                    if matched_name:
                        argument = f"vote {matched_name}"
                        print(f"✓ Voting for: {matched_name}")

            result = AgentAction(action_type=action_type, argument=argument)
        else:
            result = AgentAction(action_type="none", argument="")

        # Log player's own action to HTML
        if self.player_view and result.action_type in ["speak", "action"]:
            if result.action_type == "speak":
                self.player_view.add_event(
                    "speak", result.argument, f"{self.agent_name} (You)"
                )
            elif result.action_type == "action":
                self.player_view.add_event(
                    "action", f"You performed action: {result.argument}"
                )

        return result


def load_json(path: Path) -> Dict[str, Any]:
    return cast(Dict[str, Any], json.loads(path.read_text()))


def ensure_agent(player: Dict[str, Any]) -> AgentProfile:
    try:
        profile = AgentProfile.find(
            AgentProfile.first_name == player["first_name"],
            AgentProfile.last_name == player["last_name"],
        ).all()[0]
        return profile  # type: ignore[return-value]
    except IndexError:
        profile = AgentProfile(
            first_name=player["first_name"],
            last_name=player["last_name"],
            age=player.get("age", 30),
            occupation="",
            gender="",
            gender_pronoun=player.get("pronouns", "they/them"),
            public_info="",
            personality_and_values="",
            decision_making_style="",
            secret=player.get("secret", ""),
        )
        profile.save()
        return profile


def build_agent_goal(player: Dict[str, Any], role_prompt: str) -> str:
    return (
        f"You are {player['first_name']} {player['last_name']}, publicly known only as a villager.\n"
        f"Primary directives: {player['goal']}\n"
        f"Role guidance: {role_prompt}\n"
        f"System constraints: {COMMON_GUIDANCE}"
    )


def prepare_scenario() -> tuple[EnvironmentProfile, List[AgentProfile], Dict[str, str]]:
    role_actions = load_json(ROLE_ACTIONS_PATH)
    roster = load_json(ROSTER_PATH)

    agents: List[AgentProfile] = []
    agent_goals: List[str] = []
    role_assignments: Dict[str, str] = {}

    for player in roster["players"]:
        profile = ensure_agent(player)
        agents.append(profile)
        full_name = f"{player['first_name']} {player['last_name']}"
        role = player["role"]
        role_prompt = role_actions["roles"][role]["goal_prompt"]
        agent_goals.append(build_agent_goal(player, role_prompt))
        role_assignments[full_name] = role

    scenario_text = (
        roster["scenario"]
        + " Werewolves must be eliminated before they achieve parity with villagers."
    )

    env_profile = EnvironmentProfile(
        scenario=scenario_text,
        agent_goals=agent_goals,
        relationship=RelationshipType.acquaintance,
        game_metadata={
            "mode": "social_game",
            "rulebook_path": str(RULEBOOK_PATH),
            "actions_path": str(ROLE_ACTIONS_PATH),
            "role_assignments": role_assignments,
        },
        tag="werewolves",
    )
    env_profile.save()
    return env_profile, agents, role_assignments


def build_environment(
    env_profile: EnvironmentProfile,
    role_assignments: Dict[str, str],
    model_name: str,
) -> SocialGameEnv:
    return SocialGameEnv(
        env_profile=env_profile,
        rulebook_path=str(RULEBOOK_PATH),
        actions_path=str(ROLE_ACTIONS_PATH),
        role_assignments=role_assignments,
        model_name=model_name,
        action_order="round-robin",
        evaluators=[RuleBasedTerminatedEvaluator(max_turn_number=40, max_stale_turn=2)],
        terminal_evaluators=[
            EpisodeLLMEvaluator(
                model_name,
                EvaluationForAgents[SotopiaDimensions],
            )
        ],
    )


def create_agents(
    agent_profiles: List[AgentProfile],
    env_profile: EnvironmentProfile,
    model_names: List[str],
) -> List[Union[LLMAgent, HumanAgent]]:
    agents: List[Union[LLMAgent, HumanAgent]] = []
    for profile, model_name, goal in zip(
        agent_profiles,
        model_names,
        env_profile.agent_goals,
        strict=True,
    ):
        agent = LLMAgent(agent_profile=profile, model_name=model_name)
        agent.goal = goal
        agents.append(agent)
    return agents


def summarize_phase_log(phase_log: List[Dict[str, Any]]) -> None:
    if not phase_log:
        print("\nNo structured events recorded.")
        return

    print("\nTimeline by Phase")
    print("=" * 60)

    last_label: str | None = None
    for entry in phase_log:
        phase_name = entry["phase"]
        meta = entry.get("meta", {})
        group = meta.get("group")
        cycle = meta.get("group_cycle")
        stage = meta.get("group_stage")
        title = phase_name.replace("_", " ").title()
        if group:
            group_label = group.replace("_", " ").title()
            if cycle and stage:
                label = f"{group_label} {cycle}.{stage} – {title}"
            elif cycle:
                label = f"{group_label} {cycle} – {title}"
            else:
                label = f"{group_label}: {title}"
        else:
            label = title

        if label != last_label:
            print(f"\n[{label}]")
            last_label = label
            instructions = entry.get("instructions", [])
            for info_line in instructions:
                print(f"  Info: {info_line}")
            role_instr = entry.get("role_instructions", {})
            for role, lines in role_instr.items():
                for line in lines:
                    print(f"  Role {role}: {line}")

        for msg in entry.get("public", []):
            print(f"  Public: {msg}")
        for team, messages in entry.get("team", {}).items():
            for msg in messages:
                print(f"  Team ({team}) private: {msg}")
        for agent, messages in entry.get("private", {}).items():
            for msg in messages:
                print(f"  Private to {agent}: {msg}")
        for actor, action in entry.get("actions", {}).items():
            print(
                f"  Action logged: {actor} -> {action['action_type']} {action['argument']}"
            )


def print_roster(role_assignments: Dict[str, str]) -> None:
    print("Participants & roles:")
    for name, role in role_assignments.items():
        print(f" - {name}: {role}")


def start_http_server(port: int = 8000) -> None:
    """Start a simple HTTP server to handle player input."""
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    import threading

    class PlayerInputHandler(SimpleHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/save_action":
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode("utf-8"))
                    # Write to player_input.json in the same directory
                    input_file = BASE_DIR / "player_input.json"
                    input_file.write_text(json.dumps(data))
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(b'{"status": "success"}')
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"status": "error", "message": str(e)}).encode()
                    )
            else:
                self.send_response(404)
                self.end_headers()

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def log_message(self, _format, *_args):
            # Suppress log messages
            pass

    os.chdir(BASE_DIR)
    server = HTTPServer(("localhost", port), PlayerInputHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"✓ HTTP server started on http://localhost:{port}")


async def main() -> None:
    # Start HTTP server for handling player input from HTML
    start_http_server(8000)

    env_profile, agent_profiles, role_assignments = prepare_scenario()
    env_model = "gpt-4o-mini"
    agent_model_list = [
        "gpt-4o-mini",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "gpt-4o-mini",
    ]

    env = build_environment(env_profile, role_assignments, env_model)
    agents = create_agents(agent_profiles, env_profile, agent_model_list)

    # Get all agent names for voting support
    all_agent_names = [f"{p.first_name} {p.last_name}" for p in agent_profiles]

    # Get player info
    player_name = f"{agent_profiles[0].first_name} {agent_profiles[0].last_name}"
    player_role = list(role_assignments.values())[0]

    # Create PlayerView HTML for clean player-visible information
    player_view = PlayerView(
        PLAYER_VIEW_HTML, player_name, player_role, all_agent_names
    )

    # Replace first agent with human player that writes to PlayerView
    human_agent = PlayerViewHumanAgent(
        agent_profile=agent_profiles[0],
        available_agent_names=all_agent_names,
        player_view=player_view,
    )
    human_agent.goal = env_profile.agent_goals[0]
    agents[0] = human_agent

    print("\n🌕 Duskmire Werewolves — Interactive Social Game")
    print("=" * 60)
    print(f"You are playing as: {player_name}")
    print(f"Your role: {player_role}")
    print("=" * 60)
    print(
        "\n📖 PLAYER VIEW: Opens in your browser at http://localhost:8000/player_view.html"
    )
    print("   This shows only what your character can see + interactive input.")
    print("\n🔮 TERMINAL: Shows the full omniscient game state")
    print("   (all agent actions and decisions for debugging)")
    print("=" * 60)

    # Auto-open the HTML file in browser via HTTP server
    try:
        webbrowser.open("http://localhost:8000/player_view.html")
        print("✓ Player view opened in your browser")
    except Exception as e:
        print(f"⚠ Could not auto-open browser: {e}")
        print("   Please manually open: http://localhost:8000/player_view.html")

    print("=" * 60)
    print("Other participants:")
    for name in role_assignments.keys():
        if name != player_name:
            print(f" - {name}")
    print("=" * 60)

    await arun_one_episode(
        env=env,
        agent_list=agents,
        omniscient=False,
        script_like=False,
        json_in_script=False,
        tag=None,
        push_to_db=False,
    )

    summarize_phase_log(env.phase_log)

    if env._winner_payload:  # noqa: SLF001 (internal inspection for demo)
        print("\n" + "=" * 60)
        print("GAME RESULT")
        print("=" * 60)
        print(f"Winner: {env._winner_payload['winner']}")
        print(f"Reason: {env._winner_payload['message']}")
        player_view.add_event(
            "phase",
            f"🎮 Game Over! Winner: {env._winner_payload['winner']}. Reason: {env._winner_payload['message']}",
        )


if __name__ == "__main__":
    asyncio.run(main())
