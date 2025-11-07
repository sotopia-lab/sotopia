import asyncio
import json
import os
import random
import subprocess
import sys
import time
import typing
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, cast

import pydantic
import pytest
from fastapi import Body
from fastapi.applications import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from redis import Redis
from redis.lock import Lock
from redis_om import Migrator
from starlette.responses import Response
from pydantic import BaseModel, ConfigDict, Field

try:
    from .werewolf_state import WerewolfStateStore, action_key as werewolf_action_key
except ImportError:  # pragma: no cover
    from werewolf_state import (  # type: ignore
        WerewolfStateStore,
        action_key as werewolf_action_key,
    )

from sotopia.database import (
    AgentProfile,
    EpisodeLog,
    MatchingInWaitingRoom,
    MessageTransaction,
    SessionTransaction,
)
from pydantic import BaseModel

Migrator().run()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_URL = os.environ.get("REDIS_OM_URL", "redis://localhost:6379")
conn = Redis.from_url(REDIS_URL)
CHAT_SERVER_PATH = Path(__file__).resolve().parent / "chat_server.py"
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")
# Add to existing imports
WEREWOLF_SERVER_PATH = Path(__file__).resolve().parent / "werewolf_server.py"
WEREWOLF_STATE_PREFIX = "werewolf:session"
werewolf_state_store = WerewolfStateStore(REDIS_URL)

WAITING_ROOM_TIMEOUT = float(os.environ.get("WAITING_ROOM_TIMEOUT", 1.0))


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class WerewolfPlayer(CamelModel):
    id: str
    display_name: str
    role: str = "unknown"
    is_alive: bool = True
    is_host: bool = False


class PackMember(CamelModel):
    id: str
    display_name: str
    is_alive: bool = True
    is_human: bool = False


class PackChatMessage(CamelModel):
    phase: Optional[str] = None
    message: str
    turn: Optional[int] = None
    recorded_at: Optional[float] = None


class WerewolfPhase(CamelModel):
    phase: str
    countdown_seconds: Optional[int] = None
    description: Optional[str] = None
    allow_chat: bool = True
    allow_actions: bool = True


class WitchOptions(CamelModel):
    can_save: bool
    can_poison: bool
    pending_target: Optional[str] = None


class WerewolfSessionState(CamelModel):
    session_id: str
    players: list[WerewolfPlayer]
    me: Optional[WerewolfPlayer] = None
    phase: WerewolfPhase
    available_actions: list[str]
    last_updated: float
    status: str = "initializing"
    game_over: bool = False
    winner: Optional[str] = None
    winner_message: Optional[str] = None
    log: list[dict[str, typing.Any]] = Field(default_factory=list)
    active_player_id: Optional[str] = None
    waiting_for_action: bool = False
    host_id: Optional[str] = None
    witch_options: Optional[WitchOptions] = None
    pack_members: list[PackMember] = Field(default_factory=list)
    team_chat: list[PackChatMessage] = Field(default_factory=list)


def get_redis_client(*, decode_responses: bool = True) -> Redis:
    """Return a Redis client bound to the configured URL."""
    return Redis.from_url(
        REDIS_URL,
        decode_responses=decode_responses,
    )


@app.post("/connect/{session_id}/{role}/{id}")
async def connect(
    session_id: str, role: Literal["server", "client"], id: str
) -> list[MessageTransaction]:
    session_transactions = cast(
        list[SessionTransaction],
        SessionTransaction.find(SessionTransaction.session_id == session_id).all(),
    )
    if not session_transactions:
        if role == "client":
            raise HTTPException(status_code=404, detail="Session not found")
        else:
            session_transaction = SessionTransaction(
                session_id=session_id,
                server_id=id,
                client_id="",
                message_list=[],
            )
            session_transaction.save()
            return []
    else:
        if role == "client":
            if len(session_transactions) > 1:
                raise HTTPException(
                    status_code=500,
                    detail="Multiple session transactions found",
                )
            session_transaction = session_transactions[0]
            session_transaction.client_id = id
            session_transaction.save()
            return session_transaction.message_list
        else:
            raise HTTPException(status_code=500, detail="Session exists")


async def _get_single_exist_session(session_id: str) -> SessionTransaction:
    session_transactions = cast(
        list[SessionTransaction],
        SessionTransaction.find(SessionTransaction.session_id == session_id).all(),
    )
    if not session_transactions:
        raise HTTPException(status_code=404, detail="Session not found")
    elif len(session_transactions) > 1:
        raise HTTPException(
            status_code=500, detail="Multiple session transactions found"
        )
    else:
        return session_transactions[0]


@app.post("/send/{session_id}/{sender_id}")
async def send(
    session_id: str,
    sender_id: str,
    message: str = Body(...),
) -> list[MessageTransaction]:
    session_transaction = await _get_single_exist_session(session_id)
    sender: str = ""
    if sender_id == session_transaction.server_id:
        # Sender is server
        sender = "server"
    elif sender_id == session_transaction.client_id:
        # Sender is client
        if session_transaction.client_action_lock == "no action":
            raise HTTPException(
                status_code=412, detail="Client cannot take action now."
            )
        sender = "client"
    else:
        raise HTTPException(status_code=401, detail="Unauthorized sender")

    session_transaction.message_list.append(
        MessageTransaction(
            timestamp_str=str(datetime.now().timestamp()),
            sender=sender,
            message=message,
        )
    )
    try:
        session_transaction.save()
    except pydantic.error_wrappers.ValidationError:
        raise HTTPException(status_code=500, detail="timestamp error")
    return session_transaction.message_list


@app.put("/lock/{session_id}/{server_id}/{lock}")
async def lock(
    session_id: str, server_id: str, lock: Literal["no action", "action"]
) -> str:
    session_transaction = await _get_single_exist_session(session_id)
    if server_id != session_transaction.server_id:
        raise HTTPException(status_code=401, detail="Unauthorized sender")
    session_transaction.client_action_lock = lock
    session_transaction.save()
    return "success"


@app.get("/get/{session_id}")
async def get(session_id: str) -> list[MessageTransaction]:
    session_transaction = await _get_single_exist_session(session_id)
    return session_transaction.message_list


@app.delete("/delete/{session_id}/{server_id}")
async def delete(session_id: str, server_id: str) -> str:
    session_transaction = await _get_single_exist_session(session_id)
    if server_id != session_transaction.server_id:
        raise HTTPException(status_code=401, detail="Unauthorized sender")
    session_transaction.delete(session_transaction.pk)
    return "success"


@app.get("/get_lock/{session_id}")
async def get_lock(session_id: str) -> str:
    session_transaction = await _get_single_exist_session(session_id)
    return session_transaction.client_action_lock


def _start_server(session_ids: list[str]) -> None:
    print("start server", session_ids)
    chat_server_script = CHAT_SERVER_PATH
    if not chat_server_script.exists():
        raise RuntimeError(f"chat_server.py not found at {chat_server_script}")
    env = os.environ.copy()
    env.setdefault("FASTAPI_URL", FASTAPI_URL)
    env.setdefault("REDIS_OM_URL", os.environ.get("REDIS_OM_URL", "redis://localhost:6379"))
    subprocess.Popen(
        [
            sys.executable,
            str(chat_server_script),
            "start-server-with-session-ids",
            *session_ids,
        ],
        env=env,
    )


@app.get("/enter_waiting_room/{sender_id}")
async def enter_waiting_room(sender_id: str) -> str:
    matchings_in_waiting_room = cast(
        list[MatchingInWaitingRoom],
        MatchingInWaitingRoom.find().all(),
    )
    for matching_in_waiting_room in matchings_in_waiting_room:
        if sender_id in matching_in_waiting_room.client_ids:
            index = matching_in_waiting_room.client_ids.index(sender_id)
            match index:
                case 0:
                    if len(matching_in_waiting_room.client_ids) > 1:
                        _start_server(matching_in_waiting_room.session_ids)
                        matching_in_waiting_room.session_id_retrieved[0] = "true"
                        return matching_in_waiting_room.session_ids[0]
                    else:
                        if (
                            datetime.now().timestamp()
                            - matching_in_waiting_room.timestamp
                            > WAITING_ROOM_TIMEOUT
                        ):
                            MatchingInWaitingRoom.delete(matching_in_waiting_room.pk)
                            _start_server(matching_in_waiting_room.session_ids)
                            return matching_in_waiting_room.session_ids[0]
                        else:
                            return ""
                case 1:
                    if matching_in_waiting_room.session_id_retrieved[0]:
                        if (
                            datetime.now().timestamp()
                            - matching_in_waiting_room.timestamp
                            > WAITING_ROOM_TIMEOUT
                        ):
                            MatchingInWaitingRoom.delete(matching_in_waiting_room.pk)
                            _start_server(matching_in_waiting_room.session_ids[1:])
                            return matching_in_waiting_room.session_ids[1]
                        else:
                            return ""
                    else:
                        matching_in_waiting_room.session_id_retrieved[1] = "true"
                        MatchingInWaitingRoom.delete(matching_in_waiting_room.pk)
                        return matching_in_waiting_room.session_ids[1]
                case _:
                    assert False, f"{matching_in_waiting_room} has more than 2 clients, not expected"
    else:
        lock = Lock(conn, "lock:check_available_spots")
        with lock:
            matchings_in_waiting_room = cast(
                list[MatchingInWaitingRoom],
                MatchingInWaitingRoom.find().all(),
            )
            for matching_in_waiting_room in matchings_in_waiting_room:
                if len(matching_in_waiting_room.client_ids) == 1:
                    matching_in_waiting_room.timestamp = datetime.now().timestamp()
                    matching_in_waiting_room.client_ids.append(sender_id)
                    matching_in_waiting_room.session_ids.append(str(uuid.uuid4()))
                    matching_in_waiting_room.session_id_retrieved.append("")
                    matching_in_waiting_room.save()
                    return ""

        matching_in_waiting_room = MatchingInWaitingRoom(
            timestamp=datetime.now().timestamp(),
            client_ids=[sender_id],
            session_ids=[str(uuid.uuid4())],
            session_id_retrieved=[""],
        )
        matching_in_waiting_room.save()
        return ""


class PrettyJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=4,
            separators=(", ", ": "),
        ).encode("utf-8")


@app.get("/get_episode/{episode_id}", response_class=PrettyJSONResponse)
async def get_episode(episode_id: str) -> EpisodeLog:
    try:
        episode_log = EpisodeLog.get(pk=episode_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Episode not found: {e}")
    return episode_log


@app.get("/get_agent/{agent_id}", response_class=PrettyJSONResponse)
async def get_agent(agent_id: str) -> AgentProfile:
    try:
        agent_profile = AgentProfile.get(pk=agent_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Agent not found: {e}")
    return agent_profile

# Models (enhance existing WerewolfSessionState)
class CreateWerewolfGameRequest(BaseModel):
    host_id: str
    num_ai_players: int = 5


class WerewolfActionRequest(BaseModel):
    action_type: str
    argument: str = ""


@app.post("/games/werewolf/create")
async def create_werewolf_game(
    request: CreateWerewolfGameRequest = Body(...)
) -> dict:
    """
    Create a new werewolf game session with one human and N AI players.
    
    Returns:
        {"session_id": "...", "status": "starting", "player_name": "..."}
    """
    session_id = str(uuid.uuid4())
    
    # Create placeholder session transaction (reuse existing model)
    session_transaction = SessionTransaction(
        session_id=session_id,
        server_id=request.host_id,  # Human player ID
        client_id="",  # Not used in werewolf
        message_list=[],
    )
    session_transaction.save()
    
    # Start werewolf game server as subprocess
    if not WEREWOLF_SERVER_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"werewolf_server.py not found at {WEREWOLF_SERVER_PATH}"
        )
    
    env = os.environ.copy()
    env.setdefault("REDIS_OM_URL", os.environ.get("REDIS_OM_URL", "redis://localhost:6379"))
    
    subprocess.Popen(
        [
            sys.executable,
            str(WEREWOLF_SERVER_PATH),
            session_id,
            request.host_id,
            "--num-ai-players",
            str(request.num_ai_players),
        ],
        env=env,
    )

    # Seed initial state so the frontend has a placeholder while the game boots
    placeholder_state = WerewolfSessionState(
        session_id=session_id,
        players=[],
        phase=WerewolfPhase(
            phase="initializing",
            description="Preparing game resources…",
            allow_chat=False,
            allow_actions=False,
        ),
        available_actions=[],
        last_updated=time.time(),
        status="initializing",
        host_id=request.host_id,
        active_player_id=None,
        waiting_for_action=False,
        game_over=False,
        witch_options=None,
    )
    await werewolf_state_store.write_state(
        session_id,
        placeholder_state.model_dump(),
        ttl=60,
    )
    
    return {
        "session_id": session_id,
        "status": "starting",
        "message": "Game server launching. Poll /games/werewolf/sessions/{session_id} for state."
    }


@app.get("/games/werewolf/sessions/{session_id}", response_model=WerewolfSessionState)
async def get_werewolf_session(session_id: str) -> WerewolfSessionState:
    """
    Get current game state for a werewolf session.
    
    Frontend should poll this endpoint every 2 seconds.
    """
    state = await werewolf_state_store.read_state(session_id)

    if not state:
        raise HTTPException(
            status_code=404,
            detail="Game state not found. Game may still be initializing.",
        )

    try:
        return WerewolfSessionState(**state)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse game state: {exc}",
        ) from exc


@app.post("/games/werewolf/sessions/{session_id}/actions")
async def submit_werewolf_action(
    session_id: str,
    participant_id: str,
    action: WerewolfActionRequest = Body(...)
) -> dict:
    """
    Submit an action for the human player.
    
    The werewolf_server.py will read this from Redis and process it.
    """
    await werewolf_state_store.push_action(
        session_id=session_id,
        participant_id=participant_id,
        action_type=action.action_type,
        argument=action.argument,
    )
    
    return {
        "status": "submitted",
        "message": "Action queued for processing."
    }


@app.delete("/games/werewolf/sessions/{session_id}")
async def delete_werewolf_session(
    session_id: str,
    participant_id: str,
) -> dict:
    """Clean up game state (optional, for early exits)."""
    await werewolf_state_store.delete_state(session_id)
    await werewolf_state_store.delete_action(session_id, participant_id)
    
    # Also delete session transaction
    session_transaction = await _get_single_exist_session(session_id)
    session_transaction.delete(session_transaction.pk)
    
    return {"status": "deleted"}


client = TestClient(app)


def test_connect() -> None:
    session_id = str(uuid.uuid4())
    server_id = str(uuid.uuid4())
    response = client.post(f"/connect/{session_id}/server/{server_id}")
    assert response.status_code == 200
    assert response.json() == []

    sessions = cast(
        list[SessionTransaction],
        SessionTransaction.find(SessionTransaction.session_id == session_id).all(),
    )
    assert len(sessions) == 1
    assert sessions[0].server_id == server_id
    assert sessions[0].client_id == ""
    assert sessions[0].message_list == []
    SessionTransaction.delete(sessions[0].pk)


def test_send_message() -> None:
    session_id = str(uuid.uuid4())
    server_id = str(uuid.uuid4())
    response = client.post(f"/connect/{session_id}/server/{server_id}")
    assert response.status_code == 200
    assert response.json() == []

    response = client.post(
        f"/send/{session_id}/{server_id}",
        json="hello",
    )
    assert response.status_code == 200

    sessions = cast(
        list[SessionTransaction],
        SessionTransaction.find(SessionTransaction.session_id == session_id).all(),
    )
    assert len(sessions) == 1
    assert sessions[0].server_id == server_id
    assert sessions[0].client_id == ""
    assert len(sessions[0].message_list) == 1

    message = sessions[0].message_list[0]
    assert message.sender == "server"
    assert message.message == "hello"


@pytest.mark.asyncio
async def test_waiting_room() -> None:
    async def _join_after_seconds(
        seconds: float,
    ) -> str:
        sender_id = str(uuid.uuid4())
        await asyncio.sleep(seconds)
        while True:
            response = client.get(f"/enter_waiting_room/{sender_id}")
            if response.text:
                break
            await asyncio.sleep(0.1)
        return str(response.text)

    try:
        await asyncio.wait_for(
            asyncio.gather(
                _join_after_seconds(random.random() * 199),
                _join_after_seconds(random.random() * 199),
                _join_after_seconds(random.random() * 199),
                _join_after_seconds(random.random() * 199),
                _join_after_seconds(random.random() * 199),
            ),
            timeout=200,
        )
    except (TimeoutError, asyncio.TimeoutError) as _:
        pass
