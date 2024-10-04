# BaseSampler

`BaseSampler` is a generic class designed for sampling environments and agents in a parallel simulation framework. It provides a method to sample an environment and a list of agents based on specified parameters.

## Class Definition

```python
class BaseSampler(Generic[ObsType, ActType]):
    def __init__(
        self,
        env_candidates: Sequence[EnvironmentProfile | str] | None = None,
        agent_candidates: Sequence[AgentProfile | str] | None = None,
    ) -> None:
        self.env_candidates = env_candidates
        self.agent_candidates = agent_candidates
```

### Parameters

- `env_candidates` (`Sequence[EnvironmentProfile | str] | None`, optional): A sequence of environment profiles or strings. Defaults to `None`.
- `agent_candidates` (`Sequence[AgentProfile | str] | None`, optional): A sequence of agent profiles or strings. Defaults to `None`.

## Methods

### `sample`

```python
def sample(
    self,
    agent_classes: Type[BaseAgent[ObsType, ActType]]
    | list[Type[BaseAgent[ObsType, ActType]]],
    n_agent: int = 2,
    replacement: bool = True,
    size: int = 1,
    env_params: dict[str, Any] = {},
    agents_params: list[dict[str, Any]] = [{}, {}],
) -> Generator[EnvAgentCombo[ObsType, ActType], None, None]:
```

#### Description

Sample an environment and a list of agents.

#### Parameters

- `agent_classes` (`Type[BaseAgent[ObsType, ActType]] | list[Type[BaseAgent[ObsType, ActType]]]`): A single agent class for all sampled agents or a list of agent classes.
- `n_agent` (`int`, optional): Number of agents. Defaults to `2`.
- `replacement` (`bool`, optional): Whether to sample with replacement. Defaults to `True`.
- `size` (`int`, optional): Number of samples. Defaults to `1`.
- `env_params` (`dict[str, Any]`, optional): Parameters for the environment. Defaults to `{}`.
- `agents_params` (`list[dict[str, Any]]`, optional): Parameters for the agents. Defaults to `[{}, {}]`.

#### Returns

- `Generator[EnvAgentCombo[ObsType, ActType], None, None]`: A generator yielding tuples containing an environment and a list of agents.

## Usage Example

```python
from sotopia.agents.base_agent import BaseAgent
from sotopia.database.persistent_profile import AgentProfile, EnvironmentProfile
from sotopia.envs.parallel import ParallelSotopiaEnv

# Define a custom agent class inheriting from BaseAgent
class CustomAgent(BaseAgent):
    pass

# Initialize the BaseSampler
sampler = BaseSampler()

# Sample an environment and agents
samples = sampler.sample(agent_classes=[CustomAgent], n_agent=3, size=5)

# Iterate over the generated samples
for env, agents in samples:
    print(f"Environment: {env}")
    print(f"Agents: {agents}")
```

Note: The `sample` method raises `NotImplementedError` and must be implemented in a subclass to function correctly.
```
