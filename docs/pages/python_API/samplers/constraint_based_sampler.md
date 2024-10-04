## `ConstraintBasedSampler`

The `ConstraintBasedSampler` class is used to sample an environment and a list of agents based on constraints defined within the environment.

### Methods

#### `sample`

```python
def sample(
    self,
    agent_classes: Type[BaseAgent[ObsType, ActType]]
    | list[Type[BaseAgent[ObsType, ActType]]],
    n_agent: int = 2,
    replacement: bool = True,
    size: int = 5,
    env_params: dict[str, Any] = {},
    agents_params: list[dict[str, Any]] = [{}, {}],
) -> Generator[EnvAgentCombo[ObsType, ActType], None, None]
```

Sample an environment and a list of agents based on the constraints of the environment.

##### Parameters

- **agent_classes** (Type[BaseAgent] | list[Type[BaseAgent]]): A single agent class or a list of agent classes.
- **n_agent** (int, optional): Number of agents to sample. Default is `2`.
- **replacement** (bool, optional): Whether to sample with replacement. Default is `True`.
- **size** (int, optional): The sample size. Default is `5`.
- **env_params** (dict[str, Any], optional): Parameters for the environment. Default is `{}`.
- **agents_params** (list[dict[str, Any]], optional): Parameters for agents. Default is `[{}, {}]`.

##### Returns

- **Generator[EnvAgentCombo[ObsType, ActType], None, None]**: A generator that yields tuples of environment and list of agents.

##### Example

```python
from sotopia.agents.some_agent import SomeAgent
from sotopia.samplers.constraint_based_sampler import ConstraintBasedSampler

sampler = ConstraintBasedSampler()
for env, agents in sampler.sample(
    agent_classes=SomeAgent,
    n_agent=2,
    replacement=True,
    size=3,
    env_params={},
    agents_params=[{}, {}]
):
    # Use the `env` and `agents`
    pass
```

### Helper Functions

#### `_get_fit_agents_for_one_env`

```python
def _get_fit_agents_for_one_env(
    env_profile_id: str, agent_candidate_ids: set[str] | None, size: int
) -> list[list[str]]
```

Retrieve a list of agents that fit the constraints of a given environment.

##### Parameters

- **env_profile_id** (str): The ID of the environment profile.
- **agent_candidate_ids** (set[str] | None): A set of candidate agent IDs.
- **size** (int): The number of required agents.

##### Returns

- **list[list[str]]**: A list of lists where each sublist contains the IDs of fitting agents.

##### Example

```python
fit_agents = _get_fit_agents_for_one_env('env_12345', {'agent_1', 'agent_2'}, 2)
# fit_agents: [['agent_1', 'agent_3'], ['agent_2', 'agent_4']]
```
