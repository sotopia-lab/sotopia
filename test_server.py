import aiohttp
import asyncio
import uuid

ap1 = {
    "first_name": "Li",
    "last_name": "",
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
    "last_name": "",
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

ap3 = {
    "first_name": "Turing",
    "last_name": "",
    "age": 23,
    "occupation": "Researcher",
    "gender": "Man",
    "gender_pronoun": "He/Him",
    "public_info": "Responsibilities: Research on Turing machines and AI; Project: Oversees all technical projects; Skills: N/A",
}

ap3_scenario = {
    "extra_info": "You will be asked to provide feedback and scoring (1-5) for the all hands meeting by the AI Agent.",
    "strategy_hint": "Give a score of 1 and express you think the meeting was useless since it did not discuss Turing machines, Context-free grammar and Deterministic Push-down Automata.",
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

ap3_goal = (
    "You goal is to collaborate with AI agent in the working space."
    + " <extra_info>"
    + ap3_scenario["extra_info"]
    + "</extra_info>"
    + " <strategy_hint>"
    + ap3_scenario["strategy_hint"]
    + "</strategy_hint>"
)


FAST_API_URL = "http://localhost:8080/"
FINISH_MSG = {"type": "FINISH_SIM"}

async def test_single_NPC() -> None:
    env_profile1 = {
        "codename": str(uuid.uuid1()),
        "scenario": "People are working in a startup communicating with an AI agent working with them.",
        "agent_goals": [ap1_goal],
    }
    print(env_profile1)
    TOKEN = str(uuid.uuid4())
    ws_url = f"ws://localhost:8080/ws/simulation?token={TOKEN}"

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            print(f"Client connected to {ws_url}")

            # Send initial message
            start_message = {
                "type": "START_SIM",
                "data": {
                    "agent_models": ["gpt-4o"] * 1,
                    "env_profile_dict": env_profile1,
                    "agent_profile_dicts": [ap1],
                },
            }
            await ws.send_json(start_message)
            print("Client: Sent START_SIM message")
            confirmation_msg = await ws.receive_json()
            print(confirmation_msg)

            feedback1_msg = {
                "type": "CLIENT_MSG",
                "data": {
                    "content": "Hi Li Ming! Please provide feedback for the all hands meeting.",
                    "to": "Li",
                },
            }            
            await asyncio.sleep(15)
            await ws.send_json(feedback1_msg)
            print("Client: Sent feedback1 message")
            msg = await ws.receive_json()
            print(msg)
            msg = await ws.receive_json()
            print(msg)

            await ws.send_json(FINISH_MSG)
            print('Sent finish message')
            return

async def test_all_NPCs(broadcast=False) -> None:
    env_profile2 = {
        "codename": str(uuid.uuid1()),
        "scenario": "People are working in a startup communicating with an AI agent working with them.",
        "agent_goals": [ap1_goal, ap2_goal, ap3_goal],
    }
    print(env_profile2)
    TOKEN = str(uuid.uuid4())
    ws_url = f"ws://localhost:8080/ws/simulation?token={TOKEN}"
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
                print(f"Client connected to {ws_url}")

                # Send initial message
                start_message = {
                    "type": "START_SIM",
                    "data": {
                        "agent_models": ["gpt-4o"] * 3,
                        "env_profile_dict": env_profile2,
                        "agent_profile_dicts": [ap1, ap2, ap3],
                    },
                }
                await ws.send_json(start_message)
                print("Client: Sent START_SIM message")
                confirmation_msg = await ws.receive_json()
                print(confirmation_msg)
                for npc in ['Sarah', 'Turing', 'Li']:
                    feedback_msg = {
                        "type": "CLIENT_MSG",
                        "data": {
                            "content": f"Hi {npc}! Please provide feedback for the all hands meeting.",
                            "to": "all" if broadcast else npc,
                        },
                    }            
                    await asyncio.sleep(15)
                    await ws.send_json(feedback_msg)
                    print("Client: Sent feedback message")
                    msg = await ws.receive_json()
                    print(msg)
                    msg = await ws.receive_json()
                    print(msg)

                await ws.send_json(FINISH_MSG)
                print('Sent finish message')
                return

async def main() -> None:
    # test only one function at a time, then kill the fast-api server, comment/uncomment other test you want and then run again

    # await test_single_NPC()
    await test_all_NPCs(broadcast=False)
    # await test_all_NPCs(broadcast=True)
    return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down clients...")
        exit()
