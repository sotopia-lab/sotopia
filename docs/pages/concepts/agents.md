## Agents
Sotopia supports a variety of agents that can support social simulations for a variety of purposes.

### LLMAgent

LLMAgent is the core agent that is powered by a large language model
```python
class LLMAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "gpt-3.5-turbo",
        script_like: bool = False,
    ) -> None:
```

### ScriptWritingAgent
Script Writing Agent is an agent that is specialized in writing scripts of social interactions. It is a subclass of LLMAgent and supports the investigation of [*Is this the real life? Is this just fantasy? The Misleading Success of Simulating Social Interactions With LLMs*](https://arxiv.org/abs/2403.05020)

```python
class ScriptWritingAgent(LLMAgent):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "gpt-3.5-turbo",
        agent_names: list[str] = [],
        background: ScriptBackground | None = None,
    ) -> None:
```

### HumanAgent
This agent allows for human input to be used in the simulation through the command line.
```python
class HumanAgent(BaseAgent[Observation, AgentAction]):
    """
    A human agent that takes input from the command line.
    """

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name: LLM_Name = "human"
```

### RedisAgent

RedisAgent is an agent that can be used to interact with the Redis database.
```python
class RedisAgent(BaseAgent[Observation, AgentAction]):
    """An agent use redis as a message broker."""

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        session_id: str | None = None,
        agent_profile: AgentProfile | None = None,
    ) -> None:
```
