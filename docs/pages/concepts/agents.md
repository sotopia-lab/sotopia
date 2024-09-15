Agent is a concept in Sotopia to represent decision-making entities that can interact with each other in a social environment. Agents can be human participants, AI models, or other entities.  No matter which type of agent, they have the same interface to interact with the environment: the input is an [`Observation`](/python_API/messages/message_classes#observation) and the output is an [`AgentAction`](/python_API/messages/message_classes#agentaction), each of which is a subclass of [`Message`](/python_API/messages/message_classes#message). You can think of the environment and the agents are sending messages to each other, while the message from the environment is the observation for the agents, and the message from each of the agents is the action that they want to take in the environment.

### Actions of agents
The types of action is defined by the `ActionType` type alias, which is a literal type that can only take one of the following values: `none`, `speak`, `non-verbal communication`, `action`, `leave`. An agent can choose to n perform physical actions (`action`), use language (`speak`) or gestures or facial expressions (`non-verbal communication`) to communicate, or choose to do nothing (`none`), or leave the interaction (`leave`).

Apart from the type of action, the content of the action, e.g. the utterance, the concrete action, etc., is a free-form string in the `argument` attribute of the `AgentAction` class.


### Built-in agents
Sotopia provides several built-in agents that are ready to use. You can also create your own agents by subclassing the `BaseAgent` class.

1. [`LLMAgent`](/python_API/agents/llm_agent#llmagent): The core agent that is powered by a large language model.
2. [`HumanAgent`](/python_API/agents/llm_agent#humanagent): A command-line controlled agent that prints observation to the console and reads action from the console.
3. [`RedisAgent`](/python_API/agents/redis_agent): A RESTful API-controlled agent that sends observation to the Redis database and waits for and reads action from the Redis database. This is useful for creating a frontend interface which reads observation and sends action to the Redis database. Check out [`sotopia-chat`](https://github.com/sotopia-lab/sotopia/tree/main/sotopia-chat) for an example of how to create a chat interface using RedisAgent.
3. [`ScriptWritingAgent`](/python_API/agents/llm_agent#scriptwritingagent): An agent that has full observability of the environment.

### Creating your own agents
To create your own agents, you need to subclass the `BaseAgent` class and implement the asynchronous `aact` method. The `aact` method takes an `Observation` object as input and returns an `AgentAction` object as output. Here is an example of a simple agent that always says "Hello, world!":

```python
from sotopia.agents.base_agent import BaseAgent
from sotopia.messages.message_classes import AgentAction, Observation

class HelloWorldAgent(BaseAgent):
    async def aact(self, observation: Observation) -> AgentAction:
        return AgentAction(type="speak", argument="Hello, world!")
```
