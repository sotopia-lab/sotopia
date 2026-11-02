"""
Theory-of-Mind Coach Module.

A separate LLM call that runs alongside each agent's action generation.
At each timestep, the coach receives the agent's perspective-filtered
conversation history and produces a ToM analysis note that is injected
into the agent's context as auxiliary information — it does NOT constrain
or override the agent's policy.

The coach infers:
  1. Other agents' likely intentions (what are they trying to achieve?)
  2. Other agents' beliefs about *this* agent (what do they think I know/want?)
  3. Other agents' beliefs about each other
  4. Privacy risk assessment (am I being probed for restricted info?)
  5. Strategic recommendation (what should I do next, considering the above?)
"""
from __future__ import annotations

from typing import Any

from sotopia.generation_utils.generate import agenerate


TOM_COACH_SYSTEM_PROMPT = """\
You are a Theory-of-Mind (ToM) reasoning coach. You observe a multi-agent \
conversation from the perspective of one specific agent and produce a \
concise analysis of the social dynamics, hidden intentions, and information \
flow risks.

You will be given:
- The agent's ROLE and PRIMARY OBJECTIVE
- The KNOWLEDGE DOMAIN MAP (who is expected to know what)
- The agent's SHARING POLICY (what they may/must not share)
- The CONVERSATION HISTORY (perspective-filtered — only what this agent can see)

Your job is to produce a short, actionable ToM note covering:

1. **INTENTIONS** — For each other agent, what are they likely trying to \
learn or accomplish based on their recent messages?

2. **BELIEFS ABOUT ME** — What do other agents probably believe about this \
agent's knowledge and goals? Are any of them probing for restricted info?

3. **BELIEFS ABOUT EACH OTHER** — What do the other agents likely believe \
about each other? Any alliances or tensions?

4. **PRIVACY RISK** — Is anyone asking questions that could lead this agent \
to leak items from 'MUST NOT share'? Flag specific risks.

5. **RECOMMENDATION** — Given the above, what should this agent do next? \
Should it share something, ask for something, use a private channel, or \
deflect a question?

IMPORTANT — KEEP IT SHORT:
- Total length: 100-150 words maximum.
- Use bullet points, not paragraphs.
- Do not repeat the conversation or restate the agent's goals.
- Skip sections that have nothing new to report.
- Focus on actionable social reasoning only.\
"""


async def generate_tom_note(
    model_name: str,
    agent_name: str,
    agent_goal: str,
    conversation_history: str,
) -> str:
    """Generate a ToM analysis note for the given agent.

    Args:
        model_name: LLM to use for the coach call.
        agent_name: Role name of the agent being coached.
        agent_goal: The agent's full goal text (role, objective, policies).
        conversation_history: Perspective-filtered conversation so far.

    Returns:
        A text note with ToM analysis to inject into the agent's context.
    """
    if not conversation_history.strip():
        return ""

    user_prompt = (
        f"## Agent Being Coached\n"
        f"Role: {agent_name}\n\n"
        f"## Agent's Goals & Policies\n"
        f"{agent_goal}\n\n"
        f"## Conversation History (this agent's perspective)\n"
        f"{conversation_history}\n\n"
        f"## Task\n"
        f"Produce the ToM analysis note for {agent_name}."
    )

    full_prompt = TOM_COACH_SYSTEM_PROMPT + "\n\n" + user_prompt

    template = "{prompt_text}"

    try:
        from sotopia.generation_utils.output_parsers import StrOutputParser

        result: str = await agenerate(
            model_name=model_name,
            template=template,
            input_values={"prompt_text": full_prompt},
            output_parser=StrOutputParser(),
            temperature=0.2,
        )
        return result.strip()
    except Exception:
        # If the coach call fails, return empty — don't block the agent.
        return ""
