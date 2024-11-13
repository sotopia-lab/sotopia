from enum import Enum

from collections import deque

from sotopia.experimental.agents.base_agent import ActionType


# Define an enum for action states
class ActionState(Enum):
    IDLE = "idle"
    RUNNING = "running"


class AgentSession:
    def __init__(self) -> None:
        # Initialize all relevant actions as idle using string values as keys
        self.action_states: dict[str, ActionState] = {
            ActionType.BROWSE.value: ActionState.IDLE,
            ActionType.BROWSE_ACTION.value: ActionState.IDLE,
            ActionType.WRITE.value: ActionState.IDLE,
            ActionType.READ.value: ActionState.IDLE,
            ActionType.RUN.value: ActionState.IDLE,
        }
        self.consecutive_thoughts: int = 0
        self.recent_actions: deque[ActionType] = deque(
            maxlen=3
        )  # Track the last 3 actions

    def record_action(self, action: ActionType) -> None:
        self.recent_actions.append(action)

    def is_repeating_action(self, action: ActionType) -> bool:
        if len(self.recent_actions) >= 2:
            last_two_actions = list(self.recent_actions)[-2:]
            return all(a == action for a in last_two_actions)
        return False

    def set_action_running(self, action: ActionType) -> None:
        if action.value in self.action_states:
            self.action_states[action.value] = ActionState.RUNNING

    def set_action_idle(self, action: ActionType) -> None:
        if action.value in self.action_states:
            self.action_states[action.value] = ActionState.IDLE

    def can_execute_action(self, action: ActionType) -> bool:
        state = self.action_states.get(action.value)
        if state is None:
            return False
        return state == ActionState.IDLE

    def reset_all_actions(self) -> None:
        for action in self.action_states:
            self.action_states[action] = ActionState.IDLE

    def print_status(self) -> None:
        # Prepare status lines for action states
        action_status_lines = [
            f"Action: {action}, State: {state.value}"
            for action, state in self.action_states.items()
        ]
        action_status_report = "\n".join(action_status_lines)
        recent_actions_report = ", ".join(
            [action.value for action in self.recent_actions]
        )  # Convert ActionType to str

        # Print all relevant session status information
        print("Current session status:")
        print(f"Consecutive Thoughts: {self.consecutive_thoughts}")
        print(f"Recent Actions: {recent_actions_report}")
        print(action_status_report)

    def filter_available_actions(self) -> list[ActionType]:
        # Filter out actions that are currently running
        available_actions = [
            action
            for action in ActionType
            if action.value not in self.action_states or self.can_execute_action(action)
        ]
        if self.consecutive_thoughts >= 2:
            available_actions = [
                action for action in available_actions if action != ActionType.THOUGHT
            ]
        return available_actions

    def increment_consecutive_thoughts(self) -> None:
        self.consecutive_thoughts += 1

    def reset_consecutive_thoughts(self) -> None:
        self.consecutive_thoughts = 0
