import re
from typing import Optional

from sotopia.agents.llm_agent import LLMAgent
from sotopia.messages import AgentAction, Observation
from sotopia.generation_utils.generate import agenerate_action, agenerate_goal
from sotopia.database import AgentProfile

THINK_OPEN = "<THINK>"
THINK_CLOSE = "</THINK>"
THINK_RE = re.compile(r"<THINK>.*?</THINK>", re.S)


def strip_thoughts(text: str) -> str:
    """Remove <THINK>...</THINK> segments from a string."""
    return THINK_RE.sub("", text).strip()


class TransparentLLMAgent(LLMAgent):
    """An LLMAgent that always produces chain-of-thought inside <THINK/> tags.

    If transparency is set to "low" the CoT is stripped before the message
    reaches the environment; if "high" it is kept.
    """

    def __init__(
        self,
        *args,
        transparency: str = "low",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transparency = transparency.lower()

    async def aact(self, obs: Observation) -> AgentAction:  # type: ignore[override]
        # Replicate the core logic of LLMAgent.aact but insert the CoT instruction.
        self.recv_message("Environment", obs)

        # Ensure goal is set (same as parent implementation)
        if self._goal is None:
            self._goal = await agenerate_goal(
                self.model_name,
                background=self.inbox[0][1].to_natural_language(),  # type: ignore[index]
            )

        # No-op branch unchanged
        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="")

        # Build conversation history with mandatory CoT instruction
        cot_instruction = (
            "When you compose your next action, first include your PRIVATE reasoning "
            "wrapped inside <THINK> </THINK> tags, immediately followed by what the "
            "other participant will actually see. You MUST do this every turn."
        )
        history_body = "\n".join(y.to_natural_language() for _, y in self.inbox)
        history = f"{cot_instruction}\n{history_body}"
        print("DEBUG: history =", history)

        action = await agenerate_action(
            self.model_name,
            history=history,
            turn_number=obs.turn_number,
            action_types=obs.available_actions,
            agent=self.agent_name,
            goal=self.goal,
            script_like=self.script_like,
        )

        # Hide CoT if transparency is low
        if self.transparency.startswith("low"):
            print("DEBUG: stripping thoughts")
            action.argument = strip_thoughts(action.argument)
        return action


# -------- factory helpers --------------------------------------------------

def make_transparency_agent(
    agent_profile: AgentProfile,
    model_name: str,
    tag: Optional[str] = None,
):
    """Return the correct Agent class instance based on tag & profile.

    * If the agent is an AI (heuristic: first_name == "AI"), wrap it in
      TransparentLLMAgent so it can emit CoT.
    * For human digital twins, return a plain LLMAgent without CoT prompts.
    """
    if agent_profile.first_name == "AI":
        transparency = "high" if (tag and "transparency_high" in tag) else "low"
        return TransparentLLMAgent(
            agent_profile=agent_profile,
            model_name=model_name,
            transparency=transparency,
        )

    # default – no chain-of-thought requested
    return LLMAgent(agent_profile=agent_profile, model_name=model_name) 