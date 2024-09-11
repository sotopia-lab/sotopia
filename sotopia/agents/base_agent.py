from typing import Generic, TypeVar

from redis_om.model.model import NotFoundError

from sotopia.database import AgentProfile
from sotopia.messages import MessengerMixin

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseAgent(Generic[ObsType, ActType], MessengerMixin):
    """
    Initializes an agent with its name and profile,  provides a goal property for
    setting objectives, and defines abstract methods for taking actions (`act`)
    and asynchronous action (`aact`). The `reset` method resets the inbox, and the
    `MessengerMixin` is inherited to handle messaging functionality.

    Attributes:
        profile (AgentProfile|None): Populated based on whether a valid agent_name,
            uuid_str or agent_profile object is provided during instantiation. It
            stores information about the agent, including their name and identifier.
        agent_name (str|None): Populated from one of three sources:

            * When `agent_profile` is provided, it is set to a string concatenating
            the agent's first name and last name.
            * When `uuid_str` is provided, it attempts to fetch the corresponding
            `AgentProfile` object from the database and then sets the attribute
            in the same way as above.
            * If neither of the above conditions is met, an assertion error is
            raised unless an `agent_name` value is explicitly passed in the constructor.
        _goal (str|None): Initialized to None by default. It can be set using the
            `@property goal.setter`. It represents a goal associated with the
            agent, which must be set before it can be accessed or returned by methods.
        model_name (str): Initialized to an empty string "". It does not have a
            setter method, suggesting it's intended as an immutable attribute for
            storing the name of the agent's model.

    """

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
    ) -> None:
        """
        Initializes an agent object by setting its profile and name based on
        provided arguments. It retrieves the profile from a database or sets it
        manually if UUID or agent name is given respectively, then assigns a default
        goal and model_name.

        Args:
            agent_name (str | None): Optional by default, allowing for a custom
                agent name to be provided if neither uuid_str nor agent_profile
                is specified.
            uuid_str (str | None): Optional by default (set to None). It specifies
                an existing agent's UUID string used for retrieval from the database
                if it exists.
            agent_profile (AgentProfile | None): Used to initialize an instance
                of an agent with existing information from a database. If provided,
                it is stored as the 'profile' attribute.

        """
        MessengerMixin.__init__(self)
        if agent_profile is not None:
            self.profile = agent_profile
            self.agent_name = self.profile.first_name + " " + self.profile.last_name
        elif uuid_str is not None:
            # try retrieving profile from database
            try:
                self.profile = AgentProfile.get(pk=uuid_str)
            except NotFoundError:
                raise ValueError(f"Agent with uuid {uuid_str} not found in database")
            self.agent_name = self.profile.first_name + " " + self.profile.last_name
        else:
            assert (
                agent_name is not None
            ), "Either agent_name or uuid_str must be provided"
            self.agent_name = agent_name

        self._goal: str | None = None
        self.model_name: str = ""

    @property
    def goal(self) -> str:
        """
        Returns the goal attribute as a string, asserting that it has been set
        prior to use. This ensures that the goal is not None before attempting to
        access or return its value.

        Returns:
            str: A string representing the current goal of an object. It can be
            accessed like an attribute but is implemented as a property for
            additional functionality.

        """
        assert self._goal is not None, "attribute goal has to be set before use"
        return self._goal

    @goal.setter
    def goal(self, goal: str) -> None:
        """
        Sets and updates the agent's goal property. It takes a string argument,
        `goal`, and assigns it to the instance variable `_goal`. This allows the
        agent to store and manage its goals. The method does not return any value
        or raise exceptions.

        Args:
            goal (str): Required to be provided when calling this function because
                it has no default value specified. It represents the target or
                objective being set.

        """
        self._goal = goal

    def act(self, obs: ObsType) -> ActType:
        raise NotImplementedError

    async def aact(self, obs: ObsType) -> ActType:
        raise NotImplementedError

    def reset(self) -> None:
        self.reset_inbox()
