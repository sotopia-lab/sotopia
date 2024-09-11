import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import cast

from sotopia.agents import BaseAgent
from sotopia.database import AgentProfile
from sotopia.generation_utils.generate import (
    LLM_Name,
    agenerate_action,
    agenerate_goal,
    agenerate_script,
)
from sotopia.messages import AgentAction, Observation
from sotopia.messages.message_classes import ScriptBackground


async def ainput(prompt: str = "") -> str:
    """
    Asynchronously gets a user's input while allowing other tasks to run concurrently
    by using a thread pool executor. It returns the input as a string after stripping
    any trailing whitespace characters.

    Args:
        prompt (str): Optional; it defaults to an empty string. This means that
            if no value is provided for this parameter when calling the function,
            it will default to no prompt being shown.

    Returns:
        str: A user-input string after stripping trailing whitespace characters.

    """
    with ThreadPoolExecutor(1, "ainput") as executor:
        return (
            await asyncio.get_event_loop().run_in_executor(executor, input, prompt)
        ).rstrip()


class LLMAgent(BaseAgent[Observation, AgentAction]):
    """
    Implements an agent that interacts with an environment based on a given model
    name and goal. It receives observations, generates actions using the specified
    model, and sends messages to the environment accordingly, following a turn-based
    protocol.

    Attributes:
        model_name (str|None): Initialized by default to "gpt-3.5-turbo" with
            optional assignment in __init__ method. It appears to reference a
            specific pre-trained language model used for generating actions.
        script_like (bool): Initialized with a value of False by default. It is
            used to indicate whether the agent should generate script-like output
            or not.

    """
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "gpt-3.5-turbo",
        script_like: bool = False,
    ) -> None:
        """
        Initializes an instance with attributes for its name, unique identifier,
        profile, model type, and script-like behavior. It sets these attributes
        from provided arguments or default values if not specified.

        Args:
            agent_name (str | None): Default to None. It is set as an attribute
                of the class using super().`agent_name` = agent_name. This suggests
                that this parameter may be used for identification purposes in an
                environment with multiple agents.
            uuid_str (str | None): Optional. It defaults to None, allowing it to
                be omitted when initializing an object from this class. If provided,
                it will be passed as uuid_str to the parent class's `__init__`.
            agent_profile (AgentProfile | None): Optional.
            model_name (str): Set to "gpt-3.5-turbo" by default, but can be
                overridden with a different string value when an instance of this
                class is created.
            script_like (bool): Set to False by default. It appears to determine
                whether the object behaves like a script or not.

        """
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name = model_name
        self.script_like = script_like

    @property
    def goal(self) -> str:
        """
        Returns the goal as a string if it has been set. Otherwise, it raises an
        exception indicating that the goal is not set. The goal attribute is assumed
        to be stored as an instance variable `_goal`.

        Returns:
            str: Either a cached value stored in `_goal` attribute if it exists,
            or raises an exception indicating that goal is not set otherwise.

        """
        if self._goal is not None:
            return self._goal
        else:
            raise Exception("Goal is not set.")

    @goal.setter
    def goal(self, goal: str) -> None:
        """
        Sets the value of an instance variable `_goal` to a specified string input.
        The input is validated by type hinting as a string. The goal is then stored
        and potentially used elsewhere within the agent's logic.

        Args:
            goal (str): Used to set an instance variable named `_goal`. The `->
                None` indicates that this function does not return any value.

        """
        self._goal = goal

    def act(
        self,
        _obs: Observation,
    ) -> AgentAction:
        """
        Raises an exception when called, indicating it has been deprecated and
        replaced by the `aact` method. It is intended to be a transitional measure
        until users update their code to use the newer method.

        Args:
            _obs (Observation): Named with an underscore prefix, indicating it is
                intended to be treated as a private variable or a convention to
                indicate it is not meant to be accessed from outside this module.

        Returns:
            AgentAction: An object that represents the action taken by the agent
            based on its observations.

        """
        raise Exception("Sync act method is deprecated. Use aact instead.")

    async def aact(self, obs: Observation) -> AgentAction:
        """
        Determines an agent's next action based on observations from the environment
        and the agent's goal. It generates a response using either a pre-specified
        goal or one generated by a model, and returns it as an AgentAction object.

        Args:
            obs (Observation): An instance that represents current observation
                received from environment. It contains information about available
                actions, turn number etc.

        Returns:
            AgentAction: An object with a specific structure, containing at least
            two attributes: `action_type` and `argument`. The exact content of
            these attributes depends on various conditions within the function.

        """
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
    """
    Initializes an agent with a background script and generates actions by composing
    scripts based on observations from the environment using a language model. It
    utilizes the `agenerate_script` function to produce responses.

    Attributes:
        model_name (str|None): Initialized to "gpt-3.5-turbo" by default if not
            provided. It specifies the name of a pre-trained model used for
            generating scripts in the `agenerate_script` function call.
        agent_names (list[str]): Initialized with an empty list. It stores a list
            of names of agents involved in the interaction, which can be set by
            the user during instantiation of the agent.
        background (ScriptBackground|None): Asserted to be non-None in its
            initialization method.

    """
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "gpt-3.5-turbo",
        agent_names: list[str] = [],
        background: ScriptBackground | None = None,
    ) -> None:
        """
        Initializes an instance by setting various attributes, including model
        name and agent names, and validating that a background object is provided.
        It also calls the parent class's constructor with some of these attributes.

        Args:
            agent_name (str | None): Initialized with a default value of None. It
                represents the name of an agent, which can be assigned any string
                value or left unspecified if not provided.
            uuid_str (str | None): Initialized with a default value of None. It
                represents a string identifier, presumably a UUID (Universally
                Unique Identifier), for the agent.
            agent_profile (AgentProfile | None): Optional. It represents an agent's
                profile, which can be provided to further customize the script's
                behavior when creating a new instance.
            model_name (str): Initialized to the value `"gpt-3.5-turbo"`. This
                suggests that it specifies the name or identifier of a machine
                learning model, specifically an instance of the GPT-3.5-Turbo model.
            agent_names (list[str]): Initialized with an empty list by default.
                It is used to store names of agents, possibly multiple agents.
            background (ScriptBackground | None): Required to be non-null for
                initialization to succeed. It is used to initialize an instance
                variable named background with the provided value.

        """
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
        """
        Receives an observation, generates a script using a language model, and
        returns the generated action as an AgentAction object based on the model's
        output.

        Args:
            obs (Observation): Expected to be an instance of the class or a subclass
                of it, providing information from the environment about the current
                state or status.

        Returns:
            AgentAction: A response generated by the agent to an observation from
            the environment.

        """
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
    Simulates a human agent interacting with an environment. It retrieves available
    actions from the environment and prompts the user to select one through input.
    The selected action is then executed, allowing the agent to act based on user
    input.

    Attributes:
        model_name (LLM_Name): Initialized with a string value "human" in the
            `__init__` method. It represents the name of the model, which in this
            case is a human agent.

    """

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
    ) -> None:
        """
        Initializes an instance with agent name, UUID string, and agent profile.
        It then calls the parent class's constructor using super().__init__() and
        sets the model name to "human".

        Args:
            agent_name (str | None): Initialized with a default value of None,
                indicating that it can be either a string or omitted. It specifies
                the name of an agent being created.
            uuid_str (str | None): Optional, implying that it can take either a
                string value or be set to None when creating an instance of the class.
            agent_profile (AgentProfile | None): Optional. It represents an agent
                profile which can be used to initialize the object. The value
                passed to this parameter will be used by the parent class's constructor.

        """
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name: LLM_Name = "human"

    @property
    def goal(self) -> str:
        """
        Returns the current goal as a string. If the goal has already been set
        (i.e., _goal is not None), it simply returns the stored value. Otherwise,
        it prompts the user for input and stores the new goal in self._goal before
        returning it.

        Returns:
            str: Either a pre-existing value stored in attribute `_goal` or a
            user-entered string obtained through an interactive prompt if `_goal`
            is None.

        """
        if self._goal is not None:
            return self._goal
        goal = input("Goal: ")
        return goal

    @goal.setter
    def goal(self, goal: str) -> None:
        """
        Sets and stores a specific string value representing the agent's goal.
        This method takes one argument, a string containing the goal to be set.
        The goal is stored as an instance variable called `_goal`.

        Args:
            goal (str): A required input to be provided. It represents the objective
                or target being set. The value passed for this parameter is stored
                as an instance variable named `_goal`.

        """
        self._goal = goal

    def act(self, obs: Observation) -> AgentAction:
        """
        Receives an observation from the environment, prompts the user to select
        an action type and specify its argument, and returns an agent action based
        on the user's input.

        Args:
            obs (Observation): Passed to it when called, representing an observation
                from the environment, likely containing information about available
                actions and their arguments.

        Returns:
            AgentAction: An object containing two attributes: action_type and
            argument, both being strings that represent the selected action and
            its associated argument respectively.

        """
        self.recv_message("Environment", obs)

        print("Available actions:")
        for i, action in enumerate(obs.available_actions):
            print(f"{i}: {action}")

        action_type = obs.available_actions[int(input("Action type: "))]
        argument = input("Argument: ")

        return AgentAction(action_type=action_type, argument=argument)

    async def aact(self, obs: Observation) -> AgentAction:
        """
        Receives an observation from the environment and prompts the user to select
        an action type based on available actions. It then collects additional
        input for certain action types before returning a corresponding agent action.

        Args:
            obs (Observation): Expected to contain information about available
                actions provided by an environment.

        Returns:
            AgentAction: An object containing two attributes: `action_type` and
            `argument`. The `action_type` attribute specifies the action taken by
            the agent, and the `argument` attribute provides additional information
            about the action.

        """
        self.recv_message("Environment", obs)

        print("Available actions:")
        for i, action in enumerate(obs.available_actions):
            print(f"{i}: {action}")

        if obs.available_actions != ["none"]:
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
        if action_type in ["speak", "non-verbal communication"]:
            argument = await ainput("Argument: ")
        else:
            argument = ""

        return AgentAction(action_type=action_type, argument=argument)


class Agents(dict[str, BaseAgent[Observation, AgentAction]]):
    """
    Manages a collection of agents. It provides two methods to control and interact
    with these agents. The `reset` method allows resetting all agents in the
    collection, while the `act` method enables each agent to perform an action
    based on its observation.

    """
    def reset(self) -> None:
        """
        Iterates through all agents stored in its values and calls their respective
        `reset` methods to reset each agent's internal state. This is typically
        used to restart a simulation or experiment from a known initial condition.

        """
        for agent in self.values():
            agent.reset()

    def act(self, obs: dict[str, Observation]) -> dict[str, AgentAction]:
        """
        Aggregates actions from individual agents based on their observations. It
        iterates over each agent, calls its act method with the corresponding
        observation, and returns a dictionary mapping agent names to their respective
        actions.

        Args:
            obs (dict[str, Observation]): Populated with key-value pairs where
                keys are strings representing agent names and values are instances
                of the `Observation` class.

        Returns:
            dict[str, AgentAction]: A dictionary where keys are agent names and
            values are actions taken by agents with those names in response to
            their respective observations.

        """
        return {
            agent_name: agent.act(obs[agent_name]) for agent_name, agent in self.items()
        }
