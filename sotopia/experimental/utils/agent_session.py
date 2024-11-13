from enum import Enum

from collections import deque

from sotopia.experimental.agents.base_agent import ActionType


# Define an enum for action states
class ActionState(Enum):
    IDLE = "idle"
    RUNNING = "running"

class AgentSession:
    def __init__(self):
        # Initialize all relevant actions as idle using string values as keys
        self.action_states = {
            ActionType.BROWSE.value: ActionState.IDLE,
            ActionType.BROWSE_ACTION.value: ActionState.IDLE,
            ActionType.WRITE.value: ActionState.IDLE,
            ActionType.READ.value: ActionState.IDLE,
            ActionType.RUN.value: ActionState.IDLE,
        }
        self.consecutive_thoughts = 0
        self.recent_actions = deque(maxlen=3)  # Track the last 3 actions

    def record_action(self, action: ActionType):
        self.recent_actions.append(action)

    def is_repeating_action(self, action: ActionType) -> bool:

        # Check if the last two actions in the deque are the same as the current action
        if len(self.recent_actions) >= 2:
            last_two_actions = list(self.recent_actions)[-2:]
            return all(a == action for a in last_two_actions)

        return False
    
    def set_action_running(self, action: ActionType):
        if action.value in self.action_states:
            self.action_states[action.value] = ActionState.RUNNING


    def set_action_idle(self, action: ActionType):
        if action.value in self.action_states:
            self.action_states[action.value] = ActionState.IDLE


    def can_execute_action(self, action: ActionType) -> bool:
        state = self.action_states.get(action.value)
        if state is None:
            self.logger.warning(f"Attempted to check undefined action {action}.")
            return False
        return state == ActionState.IDLE

    def reset_all_actions(self):
        for action in self.action_states:
            self.action_states[action] = ActionState.IDLE

    def print_status(self):
        # Prepare status lines for action states
        action_status_lines = [f"Action: {action}, State: {state.value}" for action, state in self.action_states.items()]
        action_status_report = "\n".join(action_status_lines)
        recent_actions_report = ", ".join([action for action in self.recent_actions])

        # Print all relevant session status information
        print("Current session status:")
        print(f"Consecutive Thoughts: {self.consecutive_thoughts}")
        print(f"Recent Actions: {recent_actions_report}")
        print(action_status_report)
        
    def filter_available_actions(self) -> list[ActionType]:
        # Filter out actions that are currently running
        available_actions = [
            action for action in ActionType
            if action.value not in self.action_states or self.can_execute_action(action)
        ]
        if self.consecutive_thoughts >= 2:
            available_actions = [action for action in available_actions if action != ActionType.THOUGHT]
        return available_actions

    def increment_consecutive_thoughts(self):
        self.consecutive_thoughts += 1

    def reset_consecutive_thoughts(self):
        self.consecutive_thoughts = 0
      