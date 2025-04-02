# Sotopia UI
> [!CAUTION]
> Work in progress: the API endpoints are being implemented. And will be released in the future major version.

## Deploy to Modal
First you need to have a Modal account and logged in with `modal setup`

To deploy the FastAPI server to Modal, run the following command:
```bash
modal deploy scripts/modal/modal_api_server.py
```
## FastAPI Server

To run the FastAPI server, you can use the following command:
```bash
uv run rq worker
uv run fastapi run sotopia/api/fastapi_server.py --workers 4 --port 8080
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

## WebSocket Simulation

The WebSocket simulation API provides real-time interaction with simulated agents. This section explains how to use it properly.

### Connection endpoint

To connect to the WebSocket simulation API, use:
```
ws://localhost:8080/ws/simulation?token={TOKEN}
```

Where `TOKEN` is a unique identifier for your session (usually a UUID).

### Message Protocol

All messages follow the same basic structure:
```
{
    "type": "MESSAGE_TYPE",
    "data": {
        // Message-specific payload
    }
}
```

#### Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| START_SIM | Client → Server | Initialize a new simulation |
| SERVER_MSG | Server → Client | Response or update from server |
| CLIENT_MSG | Client → Server | Message from client to agents |
| ERROR | Server → Client | Error notification |
| FINISH_SIM | Client → Server | Terminate the simulation |
| END_SIM | Server → Client | Confirm simulation termination |

### Starting a Simulation

To start a simulation, send a START_SIM message with full profile information:

```json
{
    "type": "START_SIM",
    "data": {
        "agent_models": ["model1", "model2", ...],
        "env_profile_dict": {
            "pk": "unique-environment-id",
            "codename": "environment-name",
            "scenario": "Description of the scenario",
            "agent_goals": ["Goal for agent 1", "Goal for agent 2", ...]
        },
        "agent_profile_dicts": [
            {
                "pk": "unique-agent1-id",
                "first_name": "AgentName",
                "last_name": "",
                "age": 30,
                "occupation": "Developer",
                "gender": "Woman",
                "gender_pronoun": "She/Her",
                "public_info": "Description of the agent"
            },
            // Additional agents...
        ]
    }
}
```

### Sending Messages to Agents

To send a message to one or all agents, use:

```json
{
    "type": "CLIENT_MSG",
    "data": {
        "content": "Your message text here",
        "to": "AgentName"  // or "all" to broadcast
    }
}
```

### Receiving Agent Responses

When you send a CLIENT_MSG, you'll receive two messages for each agent response:
1. First message: An acknowledgment or processing message
2. Second message: The actual response from the agent

### Ending a Simulation

To end a simulation, send:

```json
{
    "type": "FINISH_SIM"
}
```

### Example Python Client

Here's a complete example of a Python client using the WebSocket API:

```python
import aiohttp
import asyncio
import uuid

async def run_simulation():
    # Define agent profiles
    agent1 = {
        "pk": f"agent-{uuid.uuid4()}",
        "first_name": "Alice",
        "last_name": "",
        "age": 28,
        "occupation": "Software Engineer",
        "gender": "Woman",
        "gender_pronoun": "She/Her",
        "public_info": "Responsibilities: Frontend development; Skills: JavaScript, React"
    }
    
    agent2 = {
        "pk": f"agent-{uuid.uuid4()}",
        "first_name": "Bob",
        "last_name": "",
        "age": 32,
        "occupation": "Product Manager",
        "gender": "Man",
        "gender_pronoun": "He/Him",
        "public_info": "Responsibilities: Feature prioritization, roadmap planning; Skills: Product strategy"
    }
    
    # Define goals for agents
    goal1 = "Your goal is to collaborate with the team. <extra_info>You need to discuss project timelines.</extra_info> <strategy_hint>Be detailed in your explanations.</strategy_hint>"
    goal2 = "Your goal is to collaborate with the team. <extra_info>You need to understand technical challenges.</extra_info> <strategy_hint>Ask specific questions.</strategy_hint>"
    
    # Define environment
    env_profile = {
        "pk": f"env-{uuid.uuid4()}",
        "codename": "team-meeting",
        "scenario": "A team meeting to discuss project status and next steps.",
        "agent_goals": [goal1, goal2]
    }
    
    # Connect to WebSocket
    TOKEN = str(uuid.uuid4())
    ws_url = f"ws://localhost:8080/ws/simulation?token={TOKEN}"
    
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            print(f"Connected to {ws_url}")
            
            # Start simulation
            start_message = {
                "type": "START_SIM",
                "data": {
                    "agent_models": ["gpt-4o", "gpt-4o"],
                    "env_profile_dict": env_profile,
                    "agent_profile_dicts": [agent1, agent2]
                }
            }
            await ws.send_json(start_message)
            confirmation = await ws.receive_json()
            print(f"Received confirmation: {confirmation}")
            
            # Send message to Alice
            client_msg = {
                "type": "CLIENT_MSG",
                "data": {
                    "content": "Hi Alice, what's your progress on the frontend tasks?",
                    "to": "Alice"
                }
            }
            await ws.send_json(client_msg)
            
            # Get responses (expect 2 epilogs: one showing the message sent, and another with the response)
            msg1 = await ws.receive_json()
            print(f"First response: {msg1}")
            msg2 = await ws.receive_json()
            print(f"Agent response: {msg2}")
            
            # Send another message to all agents
            broadcast_msg = {
                "type": "CLIENT_MSG",
                "data": {
                    "content": "What are your priorities for next week?",
                    "to": "all"
                }
            }
            await ws.send_json(broadcast_msg)
            
            # For each agent, we'll get 3 epilogs (one with the message sent, and one with each of the two agent's actions) 
            for _ in range(3):  
                response = await ws.receive_json()
                print(f"Response: {response}")
            
            # End simulation
            finish_msg = {
                "type": "FINISH_SIM"
            }
            await ws.send_json(finish_msg)
            final_msg = await ws.receive_json()
            print(f"Simulation ended: {final_msg}")

if __name__ == "__main__":
    asyncio.run(run_simulation())
```

### Error Handling

Handle errors by checking the message type:

```python
if message["type"] == "ERROR":
    error_data = message["data"]
    if "type" in error_data:
        error_type = error_data["type"]
        print(f"Error type: {error_type}")
    if "details" in error_data:
        error_details = error_data["details"]
        print(f"Error details: {error_details}")
```

Common errors include:
- SIMULATION_ISSUE: Problems with simulation initialization or execution
