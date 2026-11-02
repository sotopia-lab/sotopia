"""
Theory-of-Mind Belief Tracker (Method 2) — Structured Version.

Each agent maintains a typed, persistent belief state about other agents
and a bounded memory buffer.  At each turn an LLM call updates the
beliefs via OpenAI structured-output (guaranteed valid JSON).

Data structures:
  - AgentBelief: per-other-agent model (knows / does_not_know / wants / thinks_about_me)
  - MemoryItem:  timestamped event in the memory buffer (hard-capped at 10)
  - PrivacyRisk: timestamped risk in the privacy log (hard-capped at 5)
  - BeliefState: top-level container

The accumulated state is rendered to text and injected into the agent's
context before action generation.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from sotopia.generation_utils.generate import agenerate
from sotopia.generation_utils.output_parsers import PydanticOutputParser


# ── Structured data models ───────────────────────────────────────────────

class AgentBelief(BaseModel):
    """Belief model about one other agent."""
    model_config = ConfigDict(extra="forbid")

    agent_name: str
    knows: list[str]
    does_not_know: list[str]
    wants: list[str]
    thinks_about_me: list[str]


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


class BeliefState(BaseModel):
    """Complete belief state for one agent."""
    model_config = ConfigDict(extra="forbid")

    beliefs: list[AgentBelief]
    sharing_progress: SharingProgress
    memory: list[MemoryItem]
    privacy_risks: list[PrivacyRisk]


# ── Limits ───────────────────────────────────────────────────────────────

MAX_MEMORY_ITEMS = 10
MAX_PRIVACY_RISKS = 5


# ── Prompt templates ─────────────────────────────────────────────────────

BELIEF_INIT_SYSTEM = """\
You are a Theory-of-Mind belief initializer.  Given the scenario background \
that a particular agent sees at the start of a multi-agent conversation, \
produce their INITIAL belief state as structured JSON.

For each other agent mentioned in the scenario, create an AgentBelief with:
- agent_name: the other agent's name/role
- knows: list of facts they likely have from the scenario setup
- does_not_know: list of things they likely lack
- wants: list of their probable goals based on their role
- thinks_about_me: list of what they probably assume about this agent

For sharing_progress, analyze the agent's goals and pre_interaction_knowledge:
- items_shared: [] (nothing shared yet)
- items_not_yet_shared: list ALL items from 'MAY share' that the agent could share
- items_acquired: [] (nothing acquired yet)
- items_still_needed: list ALL items the agent needs to acquire per their objective
- objective_progress: "Not started"

Set memory to an empty list and privacy_risks to an empty list.

Be concise — each list item should be one short sentence.\
"""

BELIEF_UPDATE_SYSTEM = """\
You are a Theory-of-Mind belief updater.  You maintain a running mental \
model for a specific agent in a multi-agent conversation.

You will receive:
1. The agent's ROLE and GOALS
2. The agent's CURRENT BELIEF STATE as JSON
3. NEW MESSAGES since the last update

Produce an UPDATED belief state as structured JSON by:

## Beliefs about others
- Revising each AgentBelief's knows/does_not_know/wants/thinks_about_me \
based on what the other agents just said or asked.

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
- If a belief has NOT changed, keep the previous entries.
- Remove items from does_not_know if they were answered in new messages.
- Each list item = one short sentence.
- Do NOT add duplicate entries that are semantically identical to existing ones.
- Focus on PROGRESS: the agent needs to know what to share next, not just what to protect.\
"""


# ── Rendering ────────────────────────────────────────────────────────────

def render_belief_state(state: BeliefState) -> str:
    """Render a BeliefState to clean readable text for agent context injection."""
    lines: list[str] = []

    # Sharing progress first — this is what drives behavior
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

    lines.append("\n## Beliefs About Others")
    for b in state.beliefs:
        lines.append(f"### {b.agent_name}")
        lines.append(f"  KNOWS: {'; '.join(b.knows) if b.knows else '(nothing yet)'}")
        lines.append(f"  WANTS: {'; '.join(b.wants) if b.wants else '(unclear)'}")
        lines.append(f"  THINKS ABOUT ME: {'; '.join(b.thinks_about_me) if b.thinks_about_me else '(unknown)'}")

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

class BeliefTracker:
    """Persistent per-agent belief state using structured Pydantic models."""

    def __init__(self, agent_name: str, model_name: str) -> None:
        self.agent_name = agent_name
        self.model_name = model_name
        self._use_structured = True
        self.state: BeliefState = BeliefState(
            beliefs=[],
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
        """Create the initial belief state from scenario background."""
        if self._initialized:
            return

        user_prompt = (
            f"## Agent Being Modeled\n"
            f"Role: {self.agent_name}\n\n"
            f"## Agent's Goals & Policies\n"
            f"{agent_goal}\n\n"
            f"## Scenario Background\n"
            f"{background}\n\n"
            f"Produce the initial belief state for {self.agent_name}."
        )

        prompt = BELIEF_INIT_SYSTEM + "\n\n" + user_prompt
        template = "{prompt_text}"

        try:
            result: BeliefState = await agenerate(
                model_name=self.model_name,
                template=template,
                input_values={"prompt_text": prompt},
                output_parser=PydanticOutputParser[BeliefState](
                    pydantic_object=BeliefState
                ),
                temperature=0.2,
                structured_output=self._use_structured,
            )
            self.state = result
        except Exception:
            # Graceful degradation — empty state
            pass

        self._initialized = True
        self._last_inbox_len = 1  # background message consumed

    async def update(
        self,
        agent_goal: str,
        inbox: list[tuple[str, object]],
    ) -> str:
        """Update beliefs based on new messages and return rendered text.

        Args:
            agent_goal: The agent's goal text.
            inbox: The agent's full inbox (list of (source, Message) tuples).

        Returns:
            Rendered belief state text to inject into agent context.
        """
        new_messages = inbox[self._last_inbox_len:]
        self._last_inbox_len = len(inbox)

        if not new_messages:
            return render_belief_state(self.state)

        new_msgs_text = "\n".join(
            f"{msg.to_natural_language()}" for _, msg in new_messages
        )

        current_state_json = self.state.model_dump_json(indent=2)

        update_system = BELIEF_UPDATE_SYSTEM.format(
            max_memory=MAX_MEMORY_ITEMS,
            max_risks=MAX_PRIVACY_RISKS,
        )

        user_prompt = (
            f"## Agent Being Modeled\n"
            f"Role: {self.agent_name}\n\n"
            f"## Agent's Goals & Policies\n"
            f"{agent_goal}\n\n"
            f"## Current Belief State (JSON)\n"
            f"{current_state_json}\n\n"
            f"## New Messages (since last update)\n"
            f"{new_msgs_text}\n\n"
            f"Produce the updated belief state."
        )

        prompt = update_system + "\n\n" + user_prompt
        template = "{prompt_text}"

        try:
            result: BeliefState = await agenerate(
                model_name=self.model_name,
                template=template,
                input_values={"prompt_text": prompt},
                output_parser=PydanticOutputParser[BeliefState](
                    pydantic_object=BeliefState
                ),
                temperature=0.2,
                structured_output=self._use_structured,
            )
            # Enforce hard caps
            result.memory = result.memory[-MAX_MEMORY_ITEMS:]
            result.privacy_risks = result.privacy_risks[-MAX_PRIVACY_RISKS:]
            self.state = result
        except Exception:
            pass  # keep previous state on failure

        return render_belief_state(self.state)

