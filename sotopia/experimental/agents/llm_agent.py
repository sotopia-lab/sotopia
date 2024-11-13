import logging
import sys
from enum import Enum

from collections import deque
from rich.logging import RichHandler

from aact import Message, NodeFactory
from aact.messages import Text, Tick
from langchain.output_parsers import PydanticOutputParser
from sotopia.experimental.agents.base_agent import BaseAgent, AgentAction, ActionType
from sotopia.experimental.utils.agent_session import AgentSession
from sotopia.generation_utils import agenerate
from sotopia.generation_utils.generate import StrOutputParser

import json

# Check Python version
if sys.version_info >= (3, 11):
    pass
else:
    pass

# Configure logging
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.WARNING,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)
  
@NodeFactory.register("llm_agent")
class LLMAgent(BaseAgent[AgentAction | Tick | Text, AgentAction]):
    def __init__(
        self,
        input_text_channels: list[str],
        input_tick_channel: str,
        input_env_channels: list[str],
        output_channel: str,
        query_interval: int,
        agent_name: str,
        goal: str,
        model_name: str,
        redis_url: str,
    ):
        super().__init__(
            [
                (input_text_channel, AgentAction)
                for input_text_channel in input_text_channels
            ]
            + [
                (input_tick_channel, Tick),
            ]
            + [(input_env_channel, Text) for input_env_channel in input_env_channels],
            [(output_channel, AgentAction)],
            redis_url,
        )
        self.output_channel = output_channel
        self.query_interval = query_interval
        self.count_ticks = 0
        self.message_history: list[tuple[str, str, str]] = []
        self.name = agent_name
        self.model_name = model_name
        self.goal = goal
        self.session = AgentSession()

    async def send(self, message: AgentAction) -> None:
        if message.action_type == "speak":
            await self.r.publish(
                self.output_channel,
                Message[AgentAction](data=message).model_dump_json(),
            )

        elif message.action_type in ("browse", "browse_action", "write", "read", "run"):
            await self.r.publish(
                "Agent:Runtime",
                Message[AgentAction](data=message).model_dump_json(),
            )

    def _format_message_history(
        self, message_history: list[tuple[str, str, str]]
    ) -> str:
        ## TODO: akhatua Fix the mapping of action to be gramatically correct
        return "\n".join(
            (f"{speaker} {action} {message}")
            for speaker, action, message in message_history
        )

    def get_action_template(self, selected_actions: list[ActionType]) -> str:
        """
        Returns the action template string with selected actions.

        Args:
            selected_actions (list[ActionType]): List of ActionType enum members to include in the template.

        Returns:
            str: The action template with the selected actions.
        """
        base_template = """ You are talking to another agent.
        You are {agent_name}.\n
        {message_history}\nand you plan to {goal}.
        ## Action
        What is your next thought or action? Your response must be in JSON format.

        It must be an object, and it must contain two fields:
        * `action`, which is one of the actions below
        * `args`, which is a map of key-value pairs, specifying the arguments for that action
        """

        action_descriptions = {
            str(
                ActionType.SPEAK
            ): """`speak` - you can talk to the other agents to share information or ask them something. Arguments:
                * `content` - the message to send to the other agents (should be short)""",
            str(
                ActionType.THOUGHT
            ): """`thought` - only use this rarely to make a plan, set a goal, record your thoughts. Arguments:
                * `content` - the message you send yourself to organize your thoughts (should be short). You cannot think more than 2 turns.""",
            str(
                ActionType.NONE
            ): """`none` - you can choose not to take an action if you are waiting for some data""",
            str(
                ActionType.NON_VERBAL
            ): """`non-verbal` - you can choose to do a non verbal action
                * `content` - the non veral action you want to send to other agents. eg: smile, shrug, thumbs up""",
            str(ActionType.BROWSE): """`browse` - opens a web page. Arguments:
                * `url` - the URL to open, when you browse the web you must use `none` action until you get some information back. When you get the information back you must summarize the article and explain the article to the other agents.""",
            str(
                ActionType.BROWSE_ACTION
            ): """`browse_action` - actions you can take on a web browser
                * `command` - the command to run. You have 15 available commands. These commands must be a single string value of command
                    Options for `command`:
                        `command` = goto(url: str)
                            Description: Navigate to a url.
                            Examples:
                                goto('http://www.example.com')

                        `command` = go_back()
                            Description: Navigate to the previous page in history.
                            Examples:
                                go_back()

                        `command` = go_forward()
                            Description: Navigate to the next page in history.
                            Examples:
                                go_forward()

                        `command` = noop(wait_ms: float = 1000)
                            Description: Do nothing, and optionally wait for the given time (in milliseconds).
                            You can use this to get the current page content and/or wait for the page to load.
                            Examples:
                                noop()
                                noop(500)

                        `command` = scroll(delta_x: float, delta_y: float)
                            Description: Scroll horizontally and vertically. Amounts in pixels, positive for right or down scrolling, negative for left or up scrolling. Dispatches a wheel event.
                            Examples:
                                scroll(0, 200)
                                scroll(-50.2, -100.5)

                        `command` = fill(bid, value)
                            Description: Fill out a form field. It focuses the element and triggers an input event with the entered text. It works for <input>, <textarea> and [contenteditable] elements.
                            Examples:
                                fill('237', 'example value')
                                fill('45', 'multi-line\nexample')
                                fill('a12', 'example with "quotes"')

                        `command` = select_option(bid: str, options: str | list[str])
                            Description: Select one or multiple options in a <select> element. You can specify option value or label to select. Multiple options can be selected.
                            Examples:
                                select_option('a48', 'blue')
                                select_option('c48', ['red', 'green', 'blue'])

                        `command`= click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
                            Description: Click an element.
                            Examples:
                                click('a51')
                                click('b22', button='right')
                                click('48', button='middle', modifiers=['Shift'])

                        `command` = dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
                            Description: Double click an element.
                            Examples:
                                dblclick('12')
                                dblclick('ca42', button='right')
                                dblclick('178', button='middle', modifiers=['Shift'])

                        `command` = hover(bid: str)
                            Description: Hover over an element.
                            Examples:
                                hover('b8')

                        `command` = press(bid: str, key_comb: str)
                            Description: Focus the matching element and press a combination of keys. It accepts the logical key names that are emitted in the keyboardEvent.key property of the keyboard events: Backquote, Minus, Equal, Backslash, Backspace, Tab, Delete, Escape, ArrowDown, End, Enter, Home, Insert, PageDown, PageUp, ArrowRight, ArrowUp, F1 - F12, Digit0 - Digit9, KeyA - KeyZ, etc. You can alternatively specify a single character you'd like to produce such as "a" or "#". Following modification shortcuts are also supported: Shift, Control, Alt, Meta, ShiftLeft, ControlOrMeta. ControlOrMeta resolves to Control on Windows and Linux and to Meta on macOS.
                            Examples:
                                press('88', 'Backspace')
                                press('a26', 'ControlOrMeta+a')
                                press('a61', 'Meta+Shift+t')

                        `command` = focus(bid: str)
                            Description: Focus the matching element.
                            Examples:
                                focus('b455')

                        `command` = clear(bid: str)
                            Description: Clear the input field.
                            Examples:
                                clear('996')

                        `command` = drag_and_drop(from_bid: str, to_bid: str)
                            Description: Perform a drag & drop. Hover the element that will be dragged. Press left mouse button. Move mouse to the element that will receive the drop. Release left mouse button.
                            Examples:
                                drag_and_drop('56', '498')

                        `command`=  upload_file(bid: str, file: str | list[str])
                            Description: Click an element and wait for a "filechooser" event, then select one or multiple input files for upload. Relative file paths are resolved relative to the current working directory. An empty list clears the selected files.
                            Examples:
                                upload_file('572', '/home/user/my_receipt.pdf')
                                upload_file('63', ['/home/bob/Documents/image.jpg', '/home/bob/Documents/file.zip'])""",
            str(
                ActionType.READ
            ): """`read` - reads the content of a file. You should first check if a certain file exists. Arguments:
                * `path` - the path of the file to read""",
            str(ActionType.WRITE): """`write` - writes the content to a file. Arguments:
                * `path` - the path of the file to write
                * `content` - the content to write to the file""",
            str(
                ActionType.RUN
            ): """`run` - runs a command on the command line in a Linux shell. Arguments:
                * `command` - the command to run""",
            str(
                ActionType.LEAVE
            ): """`leave` - if your goals have been completed or abandoned, and you're absolutely certain that you've completed your task and have tested your work, use the leave action to stop working.""",
        }

        idx = 1
        selected_action_descriptions = "\n\n".join(
            f"[{idx}] {action_descriptions[str(action)]}"
            for action in selected_actions
            if str(action) in action_descriptions and (idx := idx + 1)
        )

        return (
            base_template
            + selected_action_descriptions
            + """
                You must prioritize actions that move you closer to your goal. Communicate briefly when necessary and focus on executing tasks effectively. Always consider the next actionable step to avoid unnecessary delays.
                Again, you must reply with JSON, and only with JSON.
            """
        )

    async def aact(self, message: AgentAction | Tick | Text) -> AgentAction:
        # self.session.print_status()
        match message:
            case Text(text=text):
                # TODO: arpandeepk - We need to move this to a different message case.
                if "BrowserOutputObservation" in text:
                    self.session.set_action_idle(ActionType.BROWSE)
                    self.session.set_action_idle(ActionType.BROWSE_ACTION)
                    self.message_history.append(
                        (
                            self.name,
                            "observation data",
                            "BrowserOutputObservation received.",
                        )
                    )
                    text = text.split("BrowserOutputObservation", 1)[1][
                        :50
                    ]  # TODO: arpandeepk: Fix this to be smarter

                elif "CmdOutputObservation" in text:
                    self.session.set_action_idle(ActionType.RUN)
                    self.message_history.append(
                        (
                            self.name,
                            "observation data",
                            "CmdOutputObservation received.",
                        )
                    )

                elif "FileWriteObservation" in text:
                    self.session.set_action_idle(ActionType.WRITE)
                    self.message_history.append(
                        (
                            self.name,
                            "observation data",
                            "FileWriteObservation received.",
                        )
                    )

                elif "FileReadObservation" in text:
                    self.session.set_action_idle(ActionType.WRITE)
                    self.message_history.append(
                        (
                            self.name,
                            "observation data",
                            "FileWriteObservation received.",
                        )
                    )

                elif "ErrorObservation" in text:
                    self.session.reset_all_actions()
                    self.message_history.append(
                        (
                            self.name,
                            "observation data",
                            "ErrorObservation received. Try addressing this error.",
                        )
                    )

                self.message_history.append((self.name, "observation data", text))
                return AgentAction(
                    agent_name=self.name, action_type="none", argument="", path=""
                )
            case Tick():
                self.count_ticks += 1
                if self.count_ticks % self.query_interval == 0:
                    available_actions = self.session.filter_available_actions()
                    template = self.get_action_template(available_actions)
                    agent_action = await agenerate(
                        model_name=self.model_name,
                        template=template,
                        input_values={
                            "message_history": self._format_message_history(
                                self.message_history
                            ),
                            "goal": self.goal,
                            "agent_name": self.name,
                        },
                        temperature=0.0,
                        output_parser=StrOutputParser(),
                        structured_output=False,
                    )
                    print("Action: ", self.name, " : ", str(agent_action))

                    agent_action = (
                        agent_action.replace("```", "")
                        .replace("json", "")
                        .strip('"')
                        .strip()
                    )

                    try:
                        data = json.loads(agent_action)
                        action = data["action"]

                        if self.session.is_repeating_action(action):
                            print("Action repeated too many times. Choosing another action.")
                            self.message_history.append((self.name, "status", "Avoid repeating actions. Consider a different action."))
                            return AgentAction(
                                agent_name=self.name,
                                action_type=ActionType.NONE.value,
                                argument="",
                                path="",
                            )
                            
                        self.session.record_action(action)

                        try:
                            if action == ActionType.THOUGHT.value:
                                self.session.increment_consecutive_thoughts()
                                if self.session.consecutive_thoughts > 2: 
                                    print("Too many consecutive thoughts. Choosing another action.")
                                    self.message_history.append((self.name, "status", "Consider acting on your thoughts to progress towards your goal."))
                                    return AgentAction(
                                        agent_name=self.name,
                                        action_type=ActionType.NONE.value,
                                        argument="",
                                        path="",
                                    )                                
                                content = data["args"]["content"]
                                self.message_history.append((self.name, action, content))
                                return AgentAction(
                                    agent_name=self.name,
                                    action_type=action,
                                    argument=content,
                                    path="",
                                )

                            elif action == ActionType.SPEAK.value:
                                self.session.reset_consecutive_thoughts()
                                content = data["args"]["content"]
                                self.message_history.append((self.name, action, content))
                                return AgentAction(
                                    agent_name=self.name,
                                    action_type=action,
                                    argument=content,
                                    path="",
                                )

                            elif action == ActionType.NON_VERBAL.value:
                                self.session.reset_consecutive_thoughts()
                                content = data["args"]["content"]
                                self.message_history.append((self.name, action, content))
                                return AgentAction(
                                    agent_name=self.name,
                                    action_type=action,
                                    argument=content,
                                    path="",
                                )

                            elif action == ActionType.BROWSE.value:
                                self.session.reset_consecutive_thoughts()
                                url = data["args"]["url"]
                                self.message_history.append((self.name, action, url))
                                self.session.set_action_running(ActionType.BROWSE)
                                self.session.set_action_running(ActionType.BROWSE_ACTION)
                                self.message_history.append((self.name, "status", f"Action {action} is running. Waiting for response."))
                                return AgentAction(
                                    agent_name=self.name,
                                    action_type=action,
                                    argument=url,
                                    path="",
                                )

                            elif action == ActionType.BROWSE_ACTION.value:
                                self.session.reset_consecutive_thoughts()
                                command = data["args"]["command"]
                                self.message_history.append((self.name, action, command))
                                self.session.set_action_running(ActionType.BROWSE)
                                self.session.set_action_running(ActionType.BROWSE_ACTION)
                                self.message_history.append((self.name, "status", f"Action {action} is running. Waiting for response."))
                                return AgentAction(
                                    agent_name=self.name,
                                    action_type=action,
                                    argument=command,
                                    path="",
                                )

                            elif action == ActionType.RUN.value:
                                self.session.reset_consecutive_thoughts()
                                command = data["args"]["command"]
                                self.message_history.append((self.name, action, command))
                                self.session.set_action_running(ActionType.RUN)
                                self.message_history.append((self.name, "status", f"Action {action} is running. Waiting for response."))
                                return AgentAction(
                                    agent_name=self.name,
                                    action_type=action,
                                    argument=command,
                                    path="",
                                )

                            elif action == ActionType.WRITE.value:
                                self.session.reset_consecutive_thoughts()
                                path = data["args"]["path"]
                                content = data["args"]["content"]
                                self.message_history.append((self.name, action, content))
                                self.session.set_action_running(ActionType.WRITE)
                                self.message_history.append((self.name, "status", f"Action {action} is running. Waiting for response."))
                                return AgentAction(
                                    agent_name=self.name,
                                    action_type=action,
                                    argument=content,
                                    path=path,
                                )

                            elif action == ActionType.READ.value:
                                self.session.reset_consecutive_thoughts()
                                path = data["args"]["path"]
                                self.message_history.append((self.name, action, path))
                                self.session.set_action_running(ActionType.READ)
                                self.message_history.append((self.name, "status", f"Action {action} is running. Waiting for response."))
                                return AgentAction(
                                    agent_name=self.name,
                                    action_type=action,
                                    argument="",
                                    path=path,
                                )

                            elif action == ActionType.NONE.value:
                                self.session.reset_consecutive_thoughts()
                                return AgentAction(
                                    agent_name=self.name,
                                    action_type=action,
                                    argument="",
                                    path="",
                                )
                            else:
                                print(f"Unknown action: {action}")
                        except KeyError as e:
                            print(f"Missing key in data: {e}")
                            self.message_history.append((self.name, "error", f"Missing key in data: {e}"))
                            return AgentAction(
                                agent_name=self.name,
                                action_type=ActionType.NONE.value,
                                argument="",
                                path="",
                            )
                        except Exception as e:
                            print(f"Unexpected error: {e}")
                            self.message_history.append((self.name, "error", f"Unexpected error: {e}"))
                            return AgentAction(
                                agent_name=self.name,
                                action_type=ActionType.NONE.value,
                                argument="",
                                path="",
                            )
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                else:
                    return AgentAction(
                        agent_name=self.name, action_type="none", argument="", path=""
                    )
            case AgentAction(
                agent_name=agent_name, action_type=action_type, argument=text
            ):
                # print(action_type, text)
                if action_type == ActionType.SPEAK.value:
                    self.message_history.append((agent_name, str(action_type), text))
                return AgentAction(
                    agent_name=self.name, action_type="none", argument="", path=""
                )
        raise ValueError(f"Unexpected message type: {type(message)}")
