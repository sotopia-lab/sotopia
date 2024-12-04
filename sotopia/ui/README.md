# Sotopia UI
<<<<<<< HEAD
=======
> [!CAUTION]
> Work in progress: the API endpoints are being implemented. And will be released in the future major version.
>>>>>>> main

## FastAPI Server

The API server is a FastAPI application that is used to connect the Sotopia UI to the Sotopia backend.
This could also help with other projects that need to connect to the Sotopia backend through HTTP requests.

Here are some initial design of the API server:

### Getting Data from the API Server

<<<<<<< HEAD
#### GET /get/scenarios/{scenario_id}

Get scenarios by scenario_id.
parameters:
- scenario_id: str

returns:
- scenarios: EnvironmentProfile


#### GET /get/scenarios
=======
#### GET /scenarios
>>>>>>> main

Get all scenarios.

returns:
- scenarios: list[EnvironmentProfile]

<<<<<<< HEAD
#### GET /get/scenarios/{sceanrio_tag}

Get scenarios by scenario_tag.
parameters:
- scenario_tag: str
=======
#### GET /scenarios/?get_by={id|tag}/{scenario_id|scenario_tag}

Get scenarios by scenario_tag.
parameters:
- get_by: Literal["id", "tag"]
- scenario_id: str or scenario_tag: str
>>>>>>> main
(This scenario tag could be a keyword; so people can search for scenarios by keywords)

returns:
- scenarios: list[EnvironmentProfile]

<<<<<<< HEAD
#### GET /get/agents
=======
#### GET /agents
>>>>>>> main

Get all agents.

returns:
- agents: list[AgentProfile]

<<<<<<< HEAD
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
=======
#### GET /agents/?get_by={id|gender|occupation}/{value}

Get agents by id, gender, or occupation.
parameters:
- get_by: Literal["id", "gender", "occupation"]
- value: str (agent_id, agent_gender, or agent_occupation)
>>>>>>> main

returns:
- agents: list[AgentProfile]


<<<<<<< HEAD
#### GET /get/episodes

Get all episodes.
=======
#### GET /episodes/?get_by={id|tag}/{episode_id|episode_tag}

Get episode by episode_tag.
parameters:
- get_by: Literal["id", "tag"]
- episode_id: str or episode_tag: str
>>>>>>> main

returns:
- episodes: list[Episode]

<<<<<<< HEAD
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

#### POST /post/agents/
=======

### Sending Data to the API Server

#### POST /agents/
>>>>>>> main

Send agent profile to the API server.
Request Body:
AgentProfile

returns:
- agent_id: str

<<<<<<< HEAD
#### POST /post/scenarios/
=======
#### POST /scenarios/
>>>>>>> main

Send scenario profile to the API server.
Request Body:
EnvironmentProfile

returns:
- scenario_id: str

<<<<<<< HEAD
### Initiating a new simulation episode

Not sure what's the best way to do this.
But maybe we would need a pydantic object to initiate a new episode

class SimulationEpisodeInitiation:
=======
#### DELETE /agents/{agent_id}

Delete agent profile from the API server.

returns:
- agent_id: str

#### DELETE /scenarios/{scenario_id}

Delete scenario profile from the API server.

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
>>>>>>> main
    scenario_id: str
    agent_ids: list[str]
    episode_tag: str
    models: list[str]
<<<<<<< HEAD

#### POST /post/episodes/
=======
```
>>>>>>> main

Send episode profile to the API server.
Request Body:
SimulationEpisodeInitiation

returns:
<<<<<<< HEAD
- episode_id: str


@ProKil: I think you mentioned that in the new design, message transactions are through the redis queue. So I think we should utilize there.
=======
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
>>>>>>> main
