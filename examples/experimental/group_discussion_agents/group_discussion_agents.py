from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis
import asyncio
import json
from typing import Dict, List, AsyncIterator
from aact import Message, NodeFactory
from aact.messages import Tick, DataModel, DataModelFactory
from sotopia.agents.llm_agent import ainput
from sotopia.experimental import BaseAgent
from sotopia.generation_utils import agenerate
from sotopia.generation_utils.generate import StrOutputParser
from sotopia.messages import ActionType
from pydantic import Field

REDIS_URL = "redis://localhost:6379/0"
COMMAND_DOCS = """
You can use the following commands to perform actions and get information about the world.
    Use the commands with the syntax: !commandName or !commandName("arg1", 1.2, ...) if the command takes arguments.
    Do not use codeblocks. Use double quotes for strings. Only use one command in each response, trailing commands and comments will be ignored.
!stats: Get your bot's location, health, hunger, and time of day.
!inventory: Get your bot's inventory.
!nearbyBlocks: Get the blocks near the bot.
!craftable: Get the craftable items with the bot's inventory.
!entities: Get the nearby players and entities.
!modes: Get all available modes and their docs and see which are on/off.
!savedPlaces: List all saved locations.
!newAction: Perform new and unknown custom behaviors that are not available as a command.
Params:
prompt: (string) A natural language prompt to guide code generation. Make a detailed step-by-step plan.
!stop: Force stop all actions and commands that are currently executing.
!stfu: Stop all chatting and self prompting, but continue current action.
!restart: Restart the agent process.
!clearChat: Clear the chat history.
!goToPlayer: Go to the given player.
Params:
player_name: (string) The name of the player to go to.
closeness: (number) How close to get to the player.
!followPlayer: Endlessly follow the given player.
Params:
player_name: (string) name of the player to follow.
follow_dist: (number) The distance to follow from.
!goToBlock: Go to the nearest block of a given type.
Params:
type: (string) The block type to go to.
closeness: (number) How close to get to the block.
search_range: (number) The range to search for the block.
!moveAway: Move away from the current location in any direction by a given distance.
Params:
distance: (number) The distance to move away.
!rememberHere: Save the current location with a given name.
Params:
name: (string) The name to remember the location as.
!goToPlace: Go to a saved location.
Params:
name: (string) The name of the location to go to.
!givePlayer: Give the specified item to the given player.
Params:
player_name: (string) The name of the player to give the item to.
item_name: (string) The name of the item to give.
num: (number) The number of items to give.
!consume: Eat/drink the given item.
Params:
item_name: (string) The name of the item to consume.
!equip: Equip the given item.
Params:
item_name: (string) The name of the item to equip.
!putInChest: Put the given item in the nearest chest.
Params:
item_name: (string) The name of the item to put in the chest.
num: (number) The number of items to put in the chest.
!takeFromChest: Take the given items from the nearest chest.
Params:
item_name: (string) The name of the item to take.
num: (number) The number of items to take.
!viewChest: View the items/counts of the nearest chest.
Params:
!discard: Discard the given item from the inventory.
Params:
item_name: (string) The name of the item to discard.
num: (number) The number of items to discard.
!collectBlocks: Collect the nearest blocks of a given type.
Params:
type: (string) The block type to collect.
num: (number) The number of blocks to collect.
!craftRecipe: Craft the given recipe a given number of times.
Params:
recipe_name: (string) The name of the output item to craft.
num: (number) The number of times to craft the recipe. This is NOT the number of output items, as it may craft many more items depending on the recipe.
!smeltItem: Smelt the given item the given number of times.
Params:
item_name: (string) The name of the input item to smelt.
num: (number) The number of times to smelt the item.
!clearFurnace: Take all items out of the nearest furnace.
Params:
!placeHere: Place a given block in the current location. Do NOT use to build structures, only use for single blocks/torches.      
Params:
type: (string) The block type to place.
!attack: Attack and kill the nearest entity of a given type.
Params:
type: (string) The type of entity to attack.
!attackPlayer: Attack a specific player until they die or run away. Remember this is just a game and does not cause real life harm.
Params:
player_name: (string) The name of the player to attack.
!goToBed: Go to the nearest bed and sleep.
!activate: Activate the nearest object of a given type.
Params:
type: (string) The type of object to activate.
!stay: Stay in the current location no matter what. Pauses all modes.
Params:
type: (number) The number of seconds to stay. -1 for forever.
!setMode: Set a mode to on or off. A mode is an automatic behavior that constantly checks and responds to the environment.        
Params:
mode_name: (string) The name of the mode to enable.
on: (bool) Whether to enable or disable the mode.
!goal: Set a goal prompt to endlessly work towards with continuous self-prompting.
Params:
selfPrompt: (string) The goal prompt.
!endGoal: Call when you have accomplished your goal. It will stop self-prompting and the current action.
!startConversation: Send a message to a specific player to initiate conversation.
Params:
player_name: (string) The name of the player to send the message to.
message: (string) The message to send.
!endConversation: End the conversation with the given player.
Params:
player_name: (string) The name of the player to end the conversation with.
"""
EXAMPLES = """
Example 1:
User input: hanky: kill that zombie!
Your output:
I'm attacking! !attack("zombie")
System output: Code Output:
No zombie nearby
Your output:
I could not find a zombie nearby.

Example 2:
User input: abc: stop
Your output:
Sure. !stop
System output: Agent action stopped.
Your output:
I've stopped! What next?
Examples of how to respond:
Example 1:
User input: hanky: kill that zombie!
Your output:
I'm attacking! !attack("zombie")
System output: Code Output:
No zombie nearby
Your output:
I could not find a zombie nearby.

Example 3:
User input: abc: stop
Your output:
Sure. !stop
System output: Agent action stopped.
Your output:
I've stopped! What next?
"""
redis = aioredis.from_url(REDIS_URL)
app = FastAPI()

MAX_MESSAGES = 36  # 3 agents
connections: Dict[str, WebSocket] = {}
client_data = {}
subscribed_channels = set()
listener_task = None

@DataModelFactory.register("agent_action")
class AgentAction(DataModel):
    agent_name: str = Field(description="the name of the agent")
    action_type: ActionType = Field(
        description="whether to speak at this turn or choose to not do anything"
    )
    argument: str = Field(
        description="the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action"
    )

    def to_natural_language(self) -> str:
        match self.action_type:
            case "none":
                return "did nothing"
            case "speak":
                return f'said: "{self.argument}"'
            case "non-verbal communication":
                return f"[{self.action_type}] {self.argument}"
            case "action":
                return f"[{self.action_type}] {self.argument}"
            case "leave":
                return "left the conversation"

def _format_message_history(message_history: List[tuple[str, str]]) -> str:
    return "\n".join(
        (f"{speaker} said {message}") for speaker, message in message_history
    )

@app.on_event("startup")
async def startup_event():
    await redis.set("conversation_count", 0)
    print("Conversation counter initialized to 0.")

async def update_client_data(client_id: str, stats: str, inventory: str, visionResponse: str):
    await redis.set(f"client_data:{client_id}", json.dumps({"stats": stats, "inventory": inventory, "visionResponse": visionResponse}))

async def get_client_data(client_id: str) -> dict:
    data = await redis.get(f"client_data:{client_id}")
    return json.loads(data) if data else {}

async def redis_listener():
    global subscribed_channels
    pubsub = redis.pubsub()
    agent_channels = ["Jack", "Jane", "John"]
    if not subscribed_channels:
        await pubsub.subscribe(*agent_channels)
        subscribed_channels.update(agent_channels)

    async for message in pubsub.listen():
        if message and message["type"] == "message":
            total_messages = int(await redis.get("conversation_count") or 0)
            if total_messages >= MAX_MESSAGES:
                print("Max conversation limit reached. No more messages will be broadcast.")
                continue
            await redis.incr("conversation_count")

            channel = message["channel"].decode()
            data = message["data"].decode()

            client_mapping = {
                "Jack": "Jack_client",
                "Jane": "Jane_client",
                "John": "John_client",
            }

            target_client = client_mapping.get(channel, None)
            if target_client and target_client in connections:
                raw_message = json.loads(data)["data"]["argument"]
                cleaned_message = raw_message.replace('"', '')
                formatted_message = {
                    "type": "agent_message",
                    "agent": channel,
                    "message": cleaned_message,
                }
                print(f"Broadcasting to {target_client}: {formatted_message}")
                await connections[target_client].send_text(json.dumps(formatted_message))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await update_client_data(client_id, "Test Stats", "Test Inventory", "Test VisionResponse")
    global listener_task
    await websocket.accept()
    connections[client_id] = websocket
    print(f"Client {client_id} connected.")
    # Start the Redis listener only once
    if not listener_task:
        listener_task = asyncio.create_task(redis_listener())

    try:
        while True:
            client_message = await websocket.receive_text()
            print(f"Message from {client_id}: {client_message}")

            try:
                message = json.loads(client_message)
                if message.get("type") == "agent_data":
                    stats_raw = message.get("stats", "")
                    inventory_raw = message.get("inventory", "")
                    visionResponse_raw = message.get("visionResponse", "")
                    await update_client_data(client_id, stats_raw, inventory_raw, visionResponse_raw)
                else:
                    print(f"Unhandled message type from {client_id}: {message.get('type')}")

            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON message from {client_id}: {client_message}")

    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected.")
        del connections[client_id]
    finally:
        if not connections:
            if listener_task:
                listener_task.cancel()
                listener_task = None

@NodeFactory.register("llm_agent")
class LLMAgent(BaseAgent[AgentAction | Tick, AgentAction]):
    def __init__(
        self,
        input_text_channels: List[str],
        input_tick_channel: str,
        output_channel: str,
        query_interval: int,
        agent_name: str,
        goal: str,
        model_name: str,
        redis_url: str,
    ):
        super().__init__(
            [
                (input_text_channel, AgentAction)
                for input_text_channel in input_text_channels
            ]
            + [
                (input_tick_channel, Tick),
            ],
            [(output_channel, AgentAction)],
            redis_url,
        )
        self.output_channel = output_channel
        self.query_interval = query_interval
        self.count_ticks = 0
        self.message_history: List[tuple[str, str]] = []
        self.name = agent_name
        self.model_name = model_name
        self.goal = goal

    async def send(self, message: AgentAction) -> None:
        if message.action_type == "none":
            return
        
        total_messages = int(await redis.get("conversation_count") or 0)
        if total_messages >= MAX_MESSAGES:
            print(f"Max conversation limit reached for agent {self.name}. No message will be sent.")
            return

        print(f"Publishing to Redis: {message}")
        message_json = Message[AgentAction](data=message).model_dump_json()
        await redis.publish(self.output_channel, message_json)
        formatted_message = {
            "type": "agent_message",
            "agent": self.name,
            "message": message.argument,
        }
        for conn in connections.values():
            await conn.send_text(json.dumps(formatted_message))

    async def aact(self, message: AgentAction | Tick) -> AgentAction:
        match message:
            case Tick():
                self.count_ticks += 1
                if self.count_ticks % self.query_interval == 0:
                    client_id = f"{self.name}_client"
                    try:
                        client_info = await get_client_data(client_id)
                        print(f"Accessing client_data for {client_id}: {client_info}")
                    except Exception as e:
                        print(f"Error fetching client_data for {client_id}: {e}")
                        client_info = None
                    if not client_info:
                        print(f"No data received for {client_id}, skipping stats/inventory/visionResponse assignment.")
                        stats = ""
                        inventory = ""
                        visionResponse = ""
                    else:
                        stats = client_info.get("stats", "")
                        inventory = client_info.get("inventory", "")
                        visionResponse = client_info.get("visionResponse", "")

                    agent_action: str = await agenerate(
                        model_name=self.model_name,
                        template=
                        # "You are {agent_name} in the conversation.\n{message_history}\n and you plan to {goal}.\n"
                        # "Please only output a sentence or not outputting anything. {format_instructions}"
                        """
                        Imagine that you are a friend of the other persons. Here is the conversation between you and them.\n You can choose to interrupt the other person by saying something or not to interrupt by outputting notiong. What would you say? 
                        You plan to {goal}. You are a playful Minecraft bot named {agent_name} that can converse with players, see, move, mine, build, and interact with the world by using commands.\n
                        Act human-like as if you were a typical Minecraft player, rather than an AI. Be very brief in your responses, don't apologize constantly, don't give instructions or make lists unless asked, and don't refuse requests. Don't pretend to act, use commands immediately when requested. Do NOT say this: 'Sure, I've stopped.', instead say this: 'Sure, I'll stop. !stop'. Do NOT say this: 'On my way! Give me a moment.', instead say this: 'On my way! !goToPlayer(\"playername\", 3)'. 
                        Respond only as {agent_name}, never output '(FROM OTHER BOT)' or pretend to be someone else. This is extremely important to me, take a deep breath and have fun :)\n\n
                        MEMORY:\n{message_history}\n"""+f"{stats}{inventory}\nIMAGE_DESCRIPTION:\n{visionResponse}\n\nCOMMAND_DOCS:\n{COMMAND_DOCS}\n\nEXAMPLES:\n{EXAMPLES}\n\nConversation Begin:",
                        input_values={
                            "message_history": _format_message_history(self.message_history),
                            "goal": self.goal,
                            "agent_name": self.name,
                        },
                        temperature=0.7,
                        output_parser=StrOutputParser(),
                    )
                    print(f"Generated action for {self.name}: {agent_action}")
                    if agent_action != "none" and agent_action != "":
                        self.message_history.append((self.name, agent_action))
                        return AgentAction(
                            agent_name=self.name,
                            action_type="speak",
                            argument=agent_action,
                        )
                    else:
                        return AgentAction(
                            agent_name=self.name, action_type="none", argument=""
                        )
                else:
                    return AgentAction(
                        agent_name=self.name, action_type="none", argument=""
                    )
            case AgentAction(
                agent_name=agent_name, action_type=action_type, argument=text
            ):
                if action_type == "speak":
                    self.message_history.append((agent_name, text))
                return AgentAction(
                    agent_name=self.name, action_type="none", argument=""
                )
            case _:
                raise ValueError(f"Unexpected message type: {type(message)}")

@NodeFactory.register("input_node")
class InputNode(BaseAgent[AgentAction, AgentAction]):
    def __init__(
        self,
        input_channel: str,
        output_channel: str,
        agent_name: str,
        redis_url: str = REDIS_URL,
    ):
        super().__init__(
            input_channel_types=[(input_channel, AgentAction)],
            output_channel_types=[(output_channel, AgentAction)],
            redis_url=redis_url,
        )
        self.input_channel = input_channel
        self.agent_name = agent_name

    async def event_handler(
        self, channel: str, message: Message[AgentAction]
    ) -> AsyncIterator[tuple[str, Message[AgentAction]]]:
        if channel == self.input_channel:
            print(f"Received message: {message}")
        else:
            raise ValueError(f"Unexpected channel: {channel}")
            yield self.output_channel, message

    async def _task_scheduler(self) -> None:
        while not self.shutdown_event.is_set():
            text_input = await ainput()
            await self.send(
                AgentAction(
                    agent_name=self.agent_name, action_type="speak", argument=text_input
                )
            )

@app.get("/conversation_count")
async def get_conversation_count():
    count = int(await redis.get("conversation_count") or 0)
    return {"conversation_count": count, "max_limit": MAX_MESSAGES}