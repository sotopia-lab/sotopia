# Sotopia UI
> [!CAUTION]
> Work in progress: the API endpoints are being implemented. And will be released in the future major version.

## Deploy to Modal
First you need to have a Modal account and logged in with `modal setup`

To deploy the FastAPI server to Modal, run the following command:
```bash
cd sotopia/ui/fastapi_server
modal deploy modal_api_server.py
```
## FastAPI Server

To run the FastAPI server, you can use the following command:
```bash
uv run rq worker
uv run fastapi run sotopia/ui/fastapi_server.py --workers 4 --port 8080
```

Here is also an example of using the FastAPI server:
```bash
uv run python examples/fast_api_example.py
```

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
| SERVER_MSG | Server → Client | Standard message from server (payload: `EpisodeLog`) |
| CLIENT_MSG | Client → Server | Standard message from client (payload: TBD) |
| ERROR      | Server → Client | Error notification (payload: TBD) |
| START_SIM  | Client → Server | Initialize simulation (payload: `SimulationEpisodeInitialization`) |
| END_SIM    | Client → Server | End simulation (payload: not needed) |
| FINISH_SIM | Server → Client | Terminate simulation (payload: not needed) |


**Error Type**

| Error Code | Description |
|------------|-------------|
| NOT_AUTHORIZED | Authentication failure - invalid or expired token |
| SIMULATION_ALREADY_STARTED | Attempt to start a simulation that is already active |
| SIMULATION_NOT_STARTED | Operation attempted on an inactive simulation |
| RESOURCE_NOT_FOUND | Required env_id or agent_ids not found |
| SIMULATION_ERROR |  Error occurred during simulation execution |
| SIMULATION_INTERRUPTED | The simulation is interruped |
| OTHER | Other unspecified errors |



**Implementation plan**: Currently only support LLM-LLM simulation based on [this function](https://github.com/sotopia-lab/sotopia/blob/19d39e068c3bca9246fc366e5759414f62284f93/sotopia/server.py#L108).

**SERVER_MSG payload**
The server message is a dictionary that has the following keys:
- type: str, indicates the type of the message, typically it is "messages"
- messages: Any. Typically this is the dictionary of the `EpisodeLog` for the current simulation state. (Which means the reward part could be empty)




## An example to run simulation with the API

**Get all scenarios**:
```bash
curl -X GET "http://localhost:8000/scenarios"
```

This gonna give you all the scenarios, and you can randomly pick one


**Get all agents**:
```bash
curl -X GET "http://localhost:8000/agents"
```

This gonna give you all the agents, and you can randomly pick one

**Connecting to the websocket server**:
We recommend using Python. Here is the simplist way to start a simulation and receive the results in real time:
```python
import aiohttp
import asyncio
import json

async def main():
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f'ws://{API_BASE}/ws/simulation?token={YOUR_TOKEN}') as ws:
            start_message = {
                "type": "START_SIM",
                "data": {
                    "env_id": "{ENV_ID}",
                    "agent_ids": ["{AGENT1_PK}", "{AGENT2_PK}"],
                },
            }
            await ws.send_json(start_message)

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    print(f"Received: {msg.data}")
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
```

Please check out an detailed example in `examples/experimental/websocket/websocket_test_client.py`
