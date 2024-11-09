# Sotopia UI
> [!CAUTION]
> Work in progress: the API endpoints are being implemented. And will be released in the future major version.

## FastAPI Server

The API server is a FastAPI application that is used to connect the Sotopia UI to the Sotopia backend.
This could also help with other projects that need to connect to the Sotopia backend through HTTP requests.

Here are some initial design of the API server:

### Getting Data from the API Server

#### GET /scenarios

Get all scenarios.

returns:
- scenarios: list[EnvironmentProfile]

#### GET /scenarios/?get_by={id|tag}/{scenario_id|scenario_tag}

Get scenarios by scenario_tag.
parameters:
- get_by: Literal["id", "tag"]
- scenario_id: str or scenario_tag: str
(This scenario tag could be a keyword; so people can search for scenarios by keywords)

returns:
- scenarios: list[EnvironmentProfile]

#### GET /agents

Get all agents.

returns:
- agents: list[AgentProfile]

#### GET /agents/?get_by={id|gender|occupation}/{value}

Get agents by id, gender, or occupation.
parameters:
- get_by: Literal["id", "gender", "occupation"]
- value: str (agent_id, agent_gender, or agent_occupation)

returns:
- agents: list[AgentProfile]


#### GET /episodes

Get all episodes.

returns:
- episodes: list[Episode]

#### GET /episodes/?get_by={id|tag}/{episode_id|episode_tag}

Get episode by episode_tag.
parameters:
- get_by: Literal["id", "tag"]
- episode_id: str or episode_tag: str

returns:
- episodes: list[Episode]


### Sending Data to the API Server

#### POST /agents/

Send agent profile to the API server.
Request Body:
AgentProfile

returns:
- agent_id: str

#### POST /scenarios/

Send scenario profile to the API server.
Request Body:
EnvironmentProfile

returns:
- scenario_id: str

### Updating Data in the API Server

#### PUT /agents/{agent_id}

Update agent profile in the API server.
Request Body:
AgentProfile

returns:
- agent_id: str


#### PUT /scenarios/{scenario_id}

Update scenario profile in the API server.
Request Body:
EnvironmentProfile

returns:
- scenario_id: str

### Initiating a new non-streaming simulation episode

#### POST /episodes/

```python
class SimulationEpisodeInitiation(BaseModel):
    scenario_id: str
    agent_ids: list[str]
    episode_tag: str
    models: list[str]
```

Send episode profile to the API server.
Request Body:
SimulationEpisodeInitiation

returns:
- episode_id: str (This is the id of the episode that will be used to get the episode data, saved in the redis database)

### Initiating a new interactive streaming simulation episode (this operation will open a websocket connection)

We use the websocket connection to send the simulation step-by-step results to the UI.
Please see an example protocol [here](https://claude.site/artifacts/322011f6-597f-4819-8afb-bf8137dfb56a)
