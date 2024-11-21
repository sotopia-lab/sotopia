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


### Error Code
For RESTful APIs above we have the following error codes:
| **Error Code** | **Description**                      |
|-----------------|--------------------------------------|
| **404**         | A resource is not found             |
| **403**         | The query is not authorized         |
| **500**         | Internal running error              |

### Initiating a new non-streaming simulation episode

#### POST /episodes/
[!] Currently not planning to implement
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

#### WEBSOCKET /ws/simulate/?token={token}

Parameters:
- Token: String. User authentication token. Each token maps to a unique session.

returns:
- msg: WSMessage

**WSMessage**
```json
{
    "type": "WSMessageType",
    "data": {
        // Message-specific payload
    }
}
```

**WSMessageType**
| Type | Direction   | Description |
|-----------|--------|-------------|
| SERVER_MSG | Server → Client | Standard message from server (payload: `messageForRendering` [here](https://github.com/sotopia-lab/sotopia-demo/blob/main/socialstream/rendering_utils.py) ) |
| CLIENT_MSG | Client → Server | Standard message from client (payload: Currently not needed) |
| ERROR      | Server → Client | Error notification (payload: `{"type": ERROR_TYPE, "description": DESC}`) |
| START_SIM  | Client → Server | Initialize simulation (payload: `SimulationEpisodeInitialization`) |
| END_SIM    | Client → Server | End simulation (payload: not needed) |
| FINISH_SIM | Server → Client | Terminate simulation (payload: not needed) |


**ERROR_TYPE**

| Error Code | Description |
|------------|-------------|
| NOT_AUTHORIZED | Authentication failure - invalid or expired token |
| SIMULATION_ALREADY_STARTED | Attempt to start a simulation that is already active |
| SIMULATION_NOT_STARTED | Operation attempted on an inactive simulation |
| RESOURCE_NOT_FOUND | Required env_id or agent_ids not found |
| SIMULATION_ERROR |  Error occurred during simulation execution |
| SIMULATION_INTERRUPTED | The simulation is interruped |
| OTHER | Other unspecified errors |


**Conversation Message From the Server**
The server returns messages encapsulated in a structured format which is defined as follows:

```python
class MessageForRendering(TypedDict):
    role: str # Specifies the origin of the message. Common values include "Background Info", "Environment", "{Agent Names}
    type: str # Categorizes the nature of the message. Common types include: "comment", "said", "action"
    content: str
```

**Implementation plan**: Currently only support LLM-LLM simulation based on [this function](https://github.com/sotopia-lab/sotopia/blob/19d39e068c3bca9246fc366e5759414f62284f93/sotopia/server.py#L108).
