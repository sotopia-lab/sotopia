import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import cast

from sotopia.agents import BaseAgent
from sotopia.database import AgentProfile
from sotopia.generation_utils.generate import (
    agenerate_action,
    agenerate_goal,
    agenerate_script,
)
from sotopia.messages import AgentAction, Observation
from sotopia.messages.message_classes import ScriptBackground


async def ainput(prompt: str = "") -> str:
    with ThreadPoolExecutor(1, "ainput") as executor:
        return (
            await asyncio.get_event_loop().run_in_executor(executor, input, prompt)
        ).rstrip()


class LLMAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "gpt-4o-mini",
        script_like: bool = False,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name = model_name
        self.script_like = script_like

    @property
    def goal(self) -> str:
        if self._goal is not None:
            return self._goal
        else:
            raise Exception("Goal is not set.")

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    def act(
        self,
        _obs: Observation,
    ) -> AgentAction:
        raise Exception("Sync act method is deprecated. Use aact instead.")

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        if self._goal is None:
            self._goal = await agenerate_goal(
                self.model_name,
                background=self.inbox[0][
                    1
                ].to_natural_language(),  # Only consider the first message for now
            )

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="")
        else:
            action = await agenerate_action(
                self.model_name,
                history="\n".join(f"{y.to_natural_language()}" for x, y in self.inbox),
                turn_number=obs.turn_number,
                action_types=obs.available_actions,
                agent=self.agent_name,
                goal=self.goal,
                script_like=self.script_like,
            )
            # Temporary fix for mixtral-moe model for incorrect generation format
            if "Mixtral-8x7B-Instruct-v0.1" in self.model_name:
                current_agent = self.agent_name
                if f"{current_agent}:" in action.argument:
                    print("Fixing Mixtral's generation format")
                    action.argument = action.argument.replace(f"{current_agent}: ", "")
                elif f"{current_agent} said:" in action.argument:
                    print("Fixing Mixtral's generation format")
                    action.argument = action.argument.replace(
                        f"{current_agent} said: ", ""
                    )

            return action


class ScriptWritingAgent(LLMAgent):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "gpt-4o-mini",
        agent_names: list[str] = [],
        background: ScriptBackground | None = None,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name = model_name
        self.agent_names = agent_names
        assert background is not None, "background cannot be None"
        self.background = background

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)
        message_to_compose = [y for idx, (x, y) in enumerate(self.inbox) if idx != 0]

        history = "\n".join(f"{y.to_natural_language()}" for y in message_to_compose)

        action, prompt = await agenerate_script(
            model_name=self.model_name,
            background=self.background,
            agent_names=self.agent_names,
            history=history,
            agent_name=self.agent_name,
            single_step=True,
        )
        returned_action = cast(AgentAction, action[1][0][1])
        return returned_action


class HumanAgent(BaseAgent[Observation, AgentAction]):
    """
    A human agent that takes input from the command line.
    """

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        available_agent_names: list[str] | None = None,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name = "human"
        self.available_agent_names = available_agent_names or []

    @property
    def goal(self) -> str:
        if self._goal is not None:
            return self._goal
        goal = input("Goal: ")
        return goal

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    def act(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        print("Available actions:")
        for i, action in enumerate(obs.available_actions):
            print(f"{i}: {action}")

        action_type = obs.available_actions[int(input("Action type: "))]
        argument = input("Argument: ")

        return AgentAction(action_type=action_type, argument=argument)

    def _find_matching_name(self, user_input: str) -> str | None:
        """Find a matching agent name from partial input (case-insensitive)."""
        user_input_lower = user_input.lower().strip()

        # Try exact match first
        for name in self.available_agent_names:
            if name.lower() == user_input_lower:
                return name

        # Try partial match on first name or last name
        matches = []
        for name in self.available_agent_names:
            name_parts = name.lower().split()
            if any(part.startswith(user_input_lower) for part in name_parts):
                matches.append(name)

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print("Ambiguous name. Did you mean one of these?")
            for i, match in enumerate(matches):
                print(f"  {i}: {match}")
            return None

        return None

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        # Only print if last_turn changed (avoid duplicate prompts)
        should_prompt = True
        if len(self.inbox) >= 2:
            last_obs = self.inbox[-2][1]
            if (
                isinstance(last_obs, Observation)
                and last_obs.last_turn == obs.last_turn
            ):
                should_prompt = False

        if obs.available_actions != ["none"]:
            if should_prompt:
                print("\n" + "=" * 60)
                print("YOUR TURN")
                print("=" * 60)
                print("Available actions:")
                for i, action in enumerate(obs.available_actions):
                    print(f"{i}: {action}")
            action_type_number = await ainput(
                "Action type (Please only input the number): "
            )
            try:
                action_type_number = int(action_type_number)  # type: ignore
            except TypeError:
                print("Please input a number.")
                action_type_number = await ainput(
                    "Action type (Please only input the number): "
                )
                action_type_number = int(action_type_number)  # type: ignore
            assert isinstance(action_type_number, int), "Please input a number."
            action_type = obs.available_actions[action_type_number]
        else:
            action_type = "none"

        if action_type in ["speak", "non-verbal communication", "action"]:
            argument = await ainput("Argument: ")

            # Enhanced voting support
            if action_type == "action" and argument.lower().startswith("vote"):
                # Extract the name part after "vote"
                name_part = argument[4:].strip()
                if name_part and self.available_agent_names:
                    matched_name = self._find_matching_name(name_part)
                    if matched_name:
                        argument = f"vote {matched_name}"
                        print(f"✓ Voting for: {matched_name}")
                    else:
                        print(f"⚠ Could not find player matching '{name_part}'")
                        print("Available players:")
                        for i, name in enumerate(self.available_agent_names):
                            print(f"  {i}: {name}")
                        retry = await ainput(
                            "Enter player number or name to vote for: "
                        )
                        try:
                            idx = int(retry)
                            if 0 <= idx < len(self.available_agent_names):
                                matched_name = self.available_agent_names[idx]
                                argument = f"vote {matched_name}"
                                print(f"✓ Voting for: {matched_name}")
                        except ValueError:
                            matched_name = self._find_matching_name(retry)
                            if matched_name:
                                argument = f"vote {matched_name}"
                                print(f"✓ Voting for: {matched_name}")
        else:
            argument = ""

        return AgentAction(action_type=action_type, argument=argument)


class Agents(dict[str, BaseAgent[Observation, AgentAction]]):
    def reset(self) -> None:
        for agent in self.values():
            agent.reset()

    def act(self, obs: dict[str, Observation]) -> dict[str, AgentAction]:
        return {
            agent_name: agent.act(obs[agent_name]) for agent_name, agent in self.items()
        }
