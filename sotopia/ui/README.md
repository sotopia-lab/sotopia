# Sotopia UI

## FastAPI Server

The API server is a FastAPI application that is used to connect the Sotopia UI to the Sotopia backend.
This could also help with other projects that need to connect to the Sotopia backend through HTTP requests.

Here are some initial design of the API server:

### Getting Data from the API Server

#### GET /get/scenarios/{scenario_id}

Get scenarios by scenario_id.
parameters:
- scenario_id: str

returns:
- scenarios: EnvironmentProfile


#### GET /get/scenarios

Get all scenarios.

returns:
- scenarios: list[EnvironmentProfile]

#### GET /get/scenarios/{sceanrio_tag}

Get scenarios by scenario_tag.
parameters:
- scenario_tag: str
(This scenario tag could be a keyword; so people can search for scenarios by keywords)

returns:
- scenarios: list[EnvironmentProfile]

#### GET /get/agents

Get all agents.

returns:
- agents: list[AgentProfile]

#### GET /get/agents/{agent_id}

Get agent by agent_id.
parameters:
- agent_id: str

returns:
- agent: AgentProfile

#### GET /get/agents/{agent_gender}

Get agents by agent_gender.
parameters:
- agent_gender: Literal["male", "female"]

returns:
- agents: list[AgentProfile]

#### GET /get/agents/{agent_occupation}

Get agents by agent_occupation.
parameters:
- agent_occupation: str

returns:
- agents: list[AgentProfile]


#### GET /get/episodes

Get all episodes.

returns:
- episodes: list[Episode]

#### GET /get/episodes/{episode_tag}

Get episode by episode_tag.
parameters:
- episode_tag: str

returns:
- episode: list[Episode]

#### GET /get/episodes/{episode_id}

Get episode by episode_id.
parameters:
- episode_id: str

returns:
- episode: Episode


### Sending Data to the API Server

w#### POST /post/agents/

Send agent profile to the API server.
Request Body:
AgentProfile

returns:
- agent_id: str

#### POST /post/scenarios/

Send scenario profile to the API server.
Request Body:
EnvironmentProfile

returns:
- scenario_id: str

### Updating Data in the API Server

#### PUT /put/agents/{agent_id}

Update agent profile in the API server.
Request Body:
AgentProfile

returns:
- agent_id: str


#### PUT /put/scenarios/{scenario_id}

Update scenario profile in the API server.
Request Body:
EnvironmentProfile

returns:
- scenario_id: str

### Initiating a new simulation episode

Not sure what's the best way to do this.
But maybe we would need a pydantic object to initiate a new episode

class SimulationEpisodeInitiation:
    scenario_id: str
    agent_ids: list[str]
    episode_tag: str
    models: list[str]

#### POST /post/episodes/

Send episode profile to the API server.
Request Body:
SimulationEpisodeInitiation

returns:
- episode_id: str


@ProKil: I think you mentioned that in the new design, message transactions are through the redis queue. So I think we should utilize there.
