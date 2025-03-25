import aiohttp
import asyncio
import uuid

ap1 = {
    "first_name": "Li",
    "last_name": "Ming",
    "age": 35,
    "occupation": "Database Team Project Manager",
    "gender": "Man",
    "gender_pronoun": "He/Him",
    "public_info": "Responsibilities: Managing database projects, resource coordination, ensuring timely delivery; Project: JanusGraph (Graph Database); Skills: Java, distributed systems",
}

ap1_scenario = {
    "extra_info": "You will be asked to provide feedback and scoring (1-5) for the all hands meeting by the AI Agent.",
    "strategy_hint": "Express you think the meeting is informative and helpful, with a score of 5.",
}

ap2 = {
    "first_name": "Sarah",
    "last_name": "Johnson",
    "age": 42,
    "occupation": "CTO",
    "gender": "Woman",
    "gender_pronoun": "She/Her",
    "public_info": "Responsibilities: Technical strategy planning, R&D team leadership, new technology assessment; Project: Oversees all technical projects; Skills: N/A",
}

ap2_scenario = {
    "extra_info": "You will be asked to provide feedback and scoring (1-5) for the all hands meeting by the AI Agent.",
    "strategy_hint": "Express you think the meeting is not informative and helpful, with a score of 2.",
}

ap1_goal = (
    "You goal is to collaborate with AI agent in the working space."
    + " <extra_info>"
    + ap1_scenario["extra_info"]
    + "</extra_info>"
    + " <strategy_hint>"
    + ap1_scenario["strategy_hint"]
    + "</strategy_hint>"
)

ap2_goal = (
    "You goal is to collaborate with AI agent in the working space."
    + " <extra_info>"
    + ap2_scenario["extra_info"]
    + "</extra_info>"
    + " <strategy_hint>"
    + ap2_scenario["strategy_hint"]
    + "</strategy_hint>"
)

env_profile = {
    "codename": str(uuid.uuid1()),
    "scenario": "People are working in a startup communicating with an AI agent working with them.",
    "agent_goals": [ap1_goal, ap2_goal],
}
print(env_profile)
FAST_API_URL = "http://localhost:8080/"


async def main():
    TOKEN = str(uuid.uuid4())
    WS_URL = f"ws://localhost:8080/ws/simulation?token={TOKEN}"

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(WS_URL) as ws:
            print(f"Client connected to {WS_URL}")

            # Send initial message
            # Note: You'll need to implement the logic to get agent_ids and env_id
            # This is just an example structure

            start_message = {
                "type": "START_SIM",
                "data": {
                    "agent_models": ["gpt-4o"] * 2,
                    "env_profile_dict": env_profile,
                    "agent_profile_dicts": [ap1, ap2],
                    "mode": "group",
                },
            }
            await ws.send_json(start_message)
            print("Client: Sent START_SIM message")
            confirmation_msg = await ws.receive_json()
            print(confirmation_msg)
            msg = confirmation_msg["data"]
            agent_ids = msg["agent_ids"]
            env_id = msg["env_id"]
            feedback1_msg = {
                "type": "CLIENT_MSG",
                "data": {
                    "content": "Hi Li Ming! Please provide feedback for the all hands meeting.",
                    "target_agents": [agent_ids[0]],
                },
            }
            await ws.send_json(feedback1_msg)
            print("Client: Sent feedback1 message")
            msg = await ws.receive_json()
            print(msg)

            feedback2_msg = {
                "type": "CLIENT_MSG",
                "data": {
                    "content": "Hi Sarah! Please provide feedback for the all hands meeting.",
                    "target_agents": [agent_ids[1]],
                },
            }
            await ws.send_json(feedback2_msg)
            print("Client: Sent feedback2 message")
            msg = await ws.receive_json()
            print(msg)
            # Receive and process messages
            async for msg in ws:
                print(msg)
    pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down clients...")
        exit()
