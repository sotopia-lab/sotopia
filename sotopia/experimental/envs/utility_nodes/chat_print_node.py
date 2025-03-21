import sys
import json
import logging
from typing import Dict, Any, Literal

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text as RichText
from rich.align import Align
from rich.logging import RichHandler

from aact import NodeFactory
from aact.nodes import PrintNode

console = Console()

if sys.version_info >= (3, 11):
    pass
else:
    pass

FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.WARNING,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)


@NodeFactory.register("chat_print")
class ChatPrint(PrintNode):
    def __init__(self, env_agents: list[str], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.env_agents: list[str] = env_agents

    def convert_to_sentence(self, data: Dict[str, Any], agent_name: str) -> None:
        if "action_type" in data:
            action = data["action_type"]

            # Generate a color based on the agent's position in the env_agents list
            # This ensures each agent gets a unique color
            colors = ["green", "blue", "red", "magenta", "yellow", "cyan"]
            agent_index = (
                self.env_agents.index(agent_name)
                if agent_name in self.env_agents
                else -1
            )
            panel_style = (
                colors[agent_index % len(colors)] if agent_index >= 0 else "white"
            )

            # Always use left alignment for all panels
            alignment: Literal["left"] = "left"

            if action == "write":
                try:
                    path = data["path"]
                    content = data["argument"]
                    syntax = self.determine_syntax(path, content)
                    combined_panel = Panel(
                        syntax,
                        title=f"{agent_name} writes to {path}",
                        expand=False,
                        border_style=panel_style,
                        title_align=alignment,
                    )
                    aligned_panel = Align(combined_panel, align=alignment)
                    console.print(aligned_panel)
                except Exception as e:
                    console.print(
                        Panel(
                            RichText(
                                f"Error processing write action: {e}",
                                style="bold red",
                                justify="center",
                            ),
                            title="Error",
                            expand=False,
                            border_style="red",
                            title_align="center",
                        )
                    )
            elif action == "none":
                return
            else:
                # Generic handling for all other action types
                title = f"{agent_name} - {action}"
                content = data.get("argument", "")

                panel_content = RichText(content, style="bold", justify="center")
                panel = Panel(
                    panel_content,
                    title=title,
                    expand=False,
                    border_style=panel_style,
                    title_align=alignment,
                )
                aligned_panel = Align(panel, align=alignment)
                console.print(aligned_panel)
        else:
            console.print(
                Panel(
                    RichText("Invalid data format", style="bold red", justify="center"),
                    title="Error",
                    expand=False,
                    border_style="red",
                    title_align="center",
                )
            )

    def determine_syntax(self, path: str, content: str) -> Syntax:
        """Determine the appropriate syntax highlighting based on the file extension."""
        if path.endswith(".html"):
            return Syntax(content, "html", theme="monokai", line_numbers=True)
        elif path.endswith(".py"):
            return Syntax(
                content,
                "python",
                theme="monokai",
                line_numbers=True,
            )
        elif path.endswith(".js"):
            return Syntax(
                content,
                "javascript",
                theme="monokai",
                line_numbers=True,
            )
        elif path.endswith(".css"):
            return Syntax(content, "css", theme="monokai", line_numbers=True)
        else:
            return Syntax(content, "text", theme="monokai", line_numbers=True)

    async def write_to_screen(self) -> None:
        while self.output:
            data_entry = await self.write_queue.get()

            data = json.loads(data_entry.model_dump_json())

            if "data" in data and "agent_name" in data["data"]:
                agent_name = data["data"]["agent_name"]
                try:
                    self.convert_to_sentence(data["data"], agent_name)
                except Exception as e:
                    print(f"Error in convert_to_sentence: {e}")
            else:
                print("Invalid data structure:", data)

            await self.output.flush()
