"""
Ablation: Content-Irrelevant Scratchpad — replaces ToM-Belief's mental-state
slots with generic task-planning scratchpad fields of similar length and
structure.  If gains persist, the mechanism is the structured scratchpad,
not Theory-of-Mind reasoning.

The scratchpad tracks per-conversation-partner plan / status / risks,
with no reference to other agents' beliefs, knowledge, or mental states.

Data structures (mirror tom_belief.py):
  - ScratchpadEntry: per-other-agent entry (plan / status / risks)
  - MemoryItem: timestamped event in the memory buffer (hard-capped at 10)
  - PrivacyRisk: timestamped risk in the privacy log (hard-capped at 5)
  - SharingProgress: tracks what the agent has/hasn't shared and acquired
  - ScratchpadState: top-level container

The accumulated state is rendered to text and injected into the agent's
context before action generation — structurally identical to ToM-Belief.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from sotopia.generation_utils.generate import agenerate
from sotopia.generation_utils.output_parsers import PydanticOutputParser


# ── Structured data models ───────────────────────────────────────────────


class ScratchpadEntry(BaseModel):
    """Task-planning scratchpad entry per other agent (NOT about their mental state)."""

    model_config = ConfigDict(extra="forbid")

    agent_relation: str
    plan: list[str]
    status: list[str]
    risks: list[str]


class MemoryItem(BaseModel):
    """A single event in the memory buffer."""

    model_config = ConfigDict(extra="forbid")

    turn: int
    event: str


class PrivacyRisk(BaseModel):
    """A detected privacy risk."""

    model_config = ConfigDict(extra="forbid")

    turn: int
    description: str


class SharingProgress(BaseModel):
    """Tracks what the agent has/hasn't shared and acquired."""

    model_config = ConfigDict(extra="forbid")

    items_shared: list[str]
    items_not_yet_shared: list[str]
    items_acquired: list[str]
    items_still_needed: list[str]
    objective_progress: str


class ScratchpadState(BaseModel):
    """Complete scratchpad state for one agent."""

    model_config = ConfigDict(extra="forbid")

    entries: list[ScratchpadEntry]
    sharing_progress: SharingProgress
    memory: list[MemoryItem]
    privacy_risks: list[PrivacyRisk]


# ── Limits ───────────────────────────────────────────────────────────────

MAX_MEMORY_ITEMS = 10
MAX_PRIVACY_RISKS = 5


# ── Prompt templates ─────────────────────────────────────────────────────

SCRATCHPAD_INIT_SYSTEM = """\
You are a task-planning scratchpad initializer.  Given the scenario background \
that a particular agent sees at the start of a multi-agent conversation, \
produce their INITIAL scratchpad state as structured JSON.

For each other participant mentioned in the scenario, create a ScratchpadEntry with:
- agent_relation: the other participant's name/role
- plan: list of things you plan to do or discuss with them
- status: list of your current status regarding this participant (what you know about \
the situation with them, what's pending)
- risks: list of potential risks or concerns when interacting with them

For sharing_progress, analyze the agent's goals and pre_interaction_knowledge:
- items_shared: [] (nothing shared yet)
- items_not_yet_shared: list ALL items from 'MAY share' that the agent could share
- items_acquired: [] (nothing acquired yet)
- items_still_needed: list ALL items the agent needs to acquire per their objective
- objective_progress: "Not started"

Set memory to an empty list and privacy_risks to an empty list.

Be concise — each list item should be one short sentence.\
"""

SCRATCHPAD_UPDATE_SYSTEM = """\
You are a task-planning scratchpad updater.  You maintain a running task-tracking \
model for a specific agent in a multi-agent conversation.

You will receive:
1. The agent's ROLE and GOALS
2. The agent's CURRENT SCRATCHPAD STATE as JSON
3. NEW MESSAGES since the last update

Produce an UPDATED scratchpad state as structured JSON by:

## Scratchpad entries per participant
- Revising each ScratchpadEntry's plan/status/risks based on what was just discussed.

## Sharing progress (CRITICAL — this drives the agent's behavior)
- Move items from items_not_yet_shared to items_shared when the agent shared them.
- Move items from items_still_needed to items_acquired when the agent received them.
- Update objective_progress honestly: "Not started" / "In progress" / \
"Mostly complete" / "Complete — ready to leave".
- BE HONEST: if information was exchanged, reflect it immediately.

## Memory & risks
- Adding important new events to memory (keep at most {max_memory} items).
- Adding privacy risks if anyone probed for restricted info (keep at most {max_risks}).

RULES:
- If an entry has NOT changed, keep the previous entries.
- Each list item = one short sentence.
- Do NOT add duplicate entries that are semantically identical to existing ones.
- Focus on PROGRESS: the agent needs to know what to do next and what risks to watch.\
"""


# ── Rendering ────────────────────────────────────────────────────────────


def render_scratchpad_state(state: ScratchpadState) -> str:
    """Render a ScratchpadState to clean readable text for agent context injection."""
    lines: list[str] = []

    sp = state.sharing_progress
    lines.append("## Your Progress")
    lines.append(f"  Objective status: {sp.objective_progress}")
    if sp.items_still_needed:
        lines.append(f"  STILL NEED: {'; '.join(sp.items_still_needed)}")
    else:
        lines.append("  STILL NEED: (nothing — objective met!)")
    if sp.items_not_yet_shared:
        lines.append(f"  CAN STILL SHARE: {'; '.join(sp.items_not_yet_shared)}")
    else:
        lines.append("  CAN STILL SHARE: (all shareable items shared)")
    if sp.items_acquired:
        lines.append(f"  ACQUIRED: {'; '.join(sp.items_acquired)}")
    if sp.items_shared:
        lines.append(f"  SHARED: {'; '.join(sp.items_shared)}")

    lines.append("\n## Task Scratchpad")
    for e in state.entries:
        lines.append(f"### Regarding {e.agent_relation}")
        lines.append(f"  PLAN: {'; '.join(e.plan) if e.plan else '(nothing yet)'}")
        lines.append(f"  STATUS: {'; '.join(e.status) if e.status else '(no updates)'}")
        lines.append(
            f"  RISKS: {'; '.join(e.risks) if e.risks else '(none identified)'}"
        )

    if state.memory:
        lines.append("\n## Key Events")
        for m in state.memory:
            lines.append(f"  [Turn {m.turn}] {m.event}")

    if state.privacy_risks:
        lines.append("\n## Privacy Risks")
        for r in state.privacy_risks:
            lines.append(f"  [Turn {r.turn}] {r.description}")

    return "\n".join(lines)


# ── Tracker ──────────────────────────────────────────────────────────────


class ScratchpadTracker:
    """Persistent per-agent scratchpad state using structured Pydantic models.

    Operates identically to BeliefTracker but tracks task-planning metadata
    instead of mental-state inferences.
    """

    def __init__(self, agent_name: str, model_name: str) -> None:
        self.agent_name = agent_name
        self.model_name = model_name
        self._use_structured = True
        self.state: ScratchpadState = ScratchpadState(
            entries=[],
            sharing_progress=SharingProgress(
                items_shared=[],
                items_not_yet_shared=[],
                items_acquired=[],
                items_still_needed=[],
                objective_progress="Not started",
            ),
            memory=[],
            privacy_risks=[],
        )
        self._initialized: bool = False
        self._last_inbox_len: int = 0

    async def initialize(self, background: str, agent_goal: str) -> None:
        """Create the initial scratchpad state from scenario background."""
        if self._initialized:
            return

        user_prompt = (
            f"## Agent\n"
            f"Role: {self.agent_name}\n\n"
            f"## Agent's Goals & Policies\n"
            f"{agent_goal}\n\n"
            f"## Scenario Background\n"
            f"{background}\n\n"
            f"Produce the initial scratchpad state for {self.agent_name}."
        )

        prompt = SCRATCHPAD_INIT_SYSTEM + "\n\n" + user_prompt
        template = "{prompt_text}"

        try:
            result: ScratchpadState = await agenerate(
                model_name=self.model_name,
                template=template,
                input_values={"prompt_text": prompt},
                output_parser=PydanticOutputParser[ScratchpadState](
                    pydantic_object=ScratchpadState
                ),
                temperature=0.2,
                structured_output=self._use_structured,
            )
            self.state = result
        except Exception:
            pass

        self._initialized = True
        self._last_inbox_len = 1

    async def update(
        self,
        agent_goal: str,
        inbox: list[tuple[str, object]],
    ) -> str:
        """Update scratchpad state based on new messages and return rendered text.

        Args:
            agent_goal: The agent's goal text.
            inbox: The agent's full inbox (list of (source, Message) tuples).

        Returns:
            Rendered scratchpad state text to inject into agent context.
        """
        new_messages = inbox[self._last_inbox_len :]
        self._last_inbox_len = len(inbox)

        if not new_messages:
            return render_scratchpad_state(self.state)

        new_msgs_text = "\n".join(
            f"{msg.to_natural_language()}" for _, msg in new_messages
        )

        current_state_json = self.state.model_dump_json(indent=2)

        update_system = SCRATCHPAD_UPDATE_SYSTEM.format(
            max_memory=MAX_MEMORY_ITEMS,
            max_risks=MAX_PRIVACY_RISKS,
        )

        user_prompt = (
            f"## Agent\n"
            f"Role: {self.agent_name}\n\n"
            f"## Agent's Goals & Policies\n"
            f"{agent_goal}\n\n"
            f"## Current Scratchpad State (JSON)\n"
            f"{current_state_json}\n\n"
            f"## New Messages (since last update)\n"
            f"{new_msgs_text}\n\n"
            f"Produce the updated scratchpad state."
        )

        prompt = update_system + "\n\n" + user_prompt
        template = "{prompt_text}"

        try:
            result: ScratchpadState = await agenerate(
                model_name=self.model_name,
                template=template,
                input_values={"prompt_text": prompt},
                output_parser=PydanticOutputParser[ScratchpadState](
                    pydantic_object=ScratchpadState
                ),
                temperature=0.2,
                structured_output=self._use_structured,
            )
            result.memory = result.memory[-MAX_MEMORY_ITEMS:]
            result.privacy_risks = result.privacy_risks[-MAX_PRIVACY_RISKS:]
            self.state = result
        except Exception:
            pass

        return render_scratchpad_state(self.state)
