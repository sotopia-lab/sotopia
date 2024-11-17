import websocket  # type: ignore
import json
import rel  # type: ignore
from sotopia.database import EnvironmentProfile, AgentProfile


def on_message(ws, message) -> None:  # type: ignore
    msg = json.loads(message)
    print("\nReceived message:", json.dumps(msg, indent=2))


def on_error(ws, error) -> None:  # type: ignore
    print("Error:", error)


def on_close(ws, close_status_code, close_msg) -> None:  # type: ignore
    print("Connection closed")


def on_open(ws) -> None:  # type: ignore
    agent_ids = [agent.pk for agent in AgentProfile.find().all()[:2]]
    env_id = EnvironmentProfile.find().all()[0].pk

    print("Connection established, sending START_SIM message...")
    start_message = {
        "type": "START_SIM",
        "data": {"env_id": env_id, "agent_ids": agent_ids},
    }
    ws.send(json.dumps(start_message))


if __name__ == "__main__":
    websocket.enableTrace(True)

    ws = websocket.WebSocketApp(
        "ws://localhost:8800/ws/simulation?token=test_token",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws.run_forever(dispatcher=rel)
    rel.signal(2, rel.abort)  # Ctrl+C to abort
    rel.dispatch()
