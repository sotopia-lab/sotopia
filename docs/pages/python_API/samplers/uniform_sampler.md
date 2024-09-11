# UniformSampler

The `UniformSampler` class provides functionality to sample environments and agents uniformly for simulation purposes. It extends from the `BaseSampler` class.

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

Sample an environment and `n_agent` agents.

#### Parameters:

- **agent_classes** (`Type[BaseAgent[ObsType, ActType]]` | `list[Type[BaseAgent[ObsType, ActType]]]`): The agent class or list of agent classes to be sampled.
- **n_agent** (`int`): Number of agents to sample. Defaults to 2.
- **replacement** (`bool`): Indicates if sampling is with replacement. Defaults to `True`.
- **size** (`int`): Number of samples to generate. Defaults to 1.
- **env_params** (`dict[str, Any]`): Parameters for the environment. Defaults to an empty dictionary `{}`.
- **agents_params** (`list[dict[str, Any]]`): List of parameters for each agent. Defaults to `[{}, {}]`.

#### Returns:

- **Generator[EnvAgentCombo[ObsType, ActType], None, None]**: A generator yielding tuples of environment and agents.

#### Runtime Checks:

1. If `agent_classes` is a list, it must have a length of `n_agent`.
2. `agents_params` must also be a list of length `n_agent`.
3. Uniform sampling without replacement is not supported due to inefficiencies in sequential sampling and rejection sampling.

## Usage Example

```python
from sotopia.agents.some_agent import SomeAgent
from sotopia.samplers.uniform_sampler import UniformSampler

sampler = UniformSampler()
agent_classes = [SomeAgent, SomeAgent]
agents_params = [{"param1": "value1"}, {"param2": "value2"}]

for env, agents in sampler.sample(agent_classes, n_agent=2, agents_params=agents_params):
    print(env)
    for agent in agents:
        print(agent)
```
