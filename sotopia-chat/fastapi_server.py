from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
import sys
import time
import typing
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, cast

import pydantic
import pytest
from fastapi import Body, Header
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
    )

from sotopia.database import (
    AgentProfile,
    EpisodeLog,
    MatchingInWaitingRoom,
    MessageTransaction,
    SessionTransaction,
)

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

AVAILABLE_GAMES: dict[str, dict[str, typing.Any]] = {
    "werewolf": {
        "title": "Werewolf",
        "est_duration": 900,
        "max_parallel_sessions": 4,
    },
    "secret-mafia": {
        "title": "Secret Mafia",
        "est_duration": 1500,
        "max_parallel_sessions": 2,
    },
    "codenames-duo": {
        "title": "Codenames Duo",
        "est_duration": 720,
        "max_parallel_sessions": 1,
    },
    "colonel-blotto": {
        "title": "Colonel Blotto",
        "est_duration": 480,
        "max_parallel_sessions": 2,
    },
}

QUEUE_LOCK = asyncio.Lock()
MATCHMAKING_QUEUE: dict[str, list[str]] = {slug: [] for slug in AVAILABLE_GAMES}
MATCHMAKING_TICKETS: dict[str, dict[str, typing.Any]] = {}
MATCHMAKING_ASSIGNMENTS: deque[dict[str, typing.Any]] = deque(maxlen=50)
RECENT_WAIT_TIMES: deque[float] = deque(maxlen=50)
GLOBAL_MATCHMAKING_STATE: dict[str, typing.Any] = {
    "server_status": "online",
    "issues": [
        "Model 'Humanity' is pending approval for Stage 2 of Mind Games Challenge."
    ],
    "active_sessions": 3,
    "last_updated": time.time(),
}
SCHEDULER_TASK: asyncio.Task | None = None
TICKET_KEY_PREFIX = "arena:ticket:"
QUEUE_SNAPSHOT_KEY = "arena:queue_snapshot"
TELEMETRY_KEY = "arena:telemetry"
MATCH_LOG_KEY = "arena:match_logs"
LEADERBOARD_KEY = "arena:leaderboard"
MAX_MATCH_LOGS = 500
IDENTITY_TOKEN_KEY = "arena:identity:token:"
IDENTITY_PARTICIPANT_KEY = "arena:identity:participants"
MEMORY_KEY_PREFIX = "arena:memory:"
GAME_STATE_KEY = "arena:game_state"
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")
START_TIME = time.time()


def _cleanup_expired_tickets(now: float | None = None) -> None:
    now = now or time.time()
    expire_threshold = now - 600
    stale_ids: list[str] = []
    for ticket_id, record in MATCHMAKING_TICKETS.items():
        if record["status"] == "queued" and record["queued_at"] < expire_threshold:
            stale_ids.append(ticket_id)
    for ticket_id in stale_ids:
        MATCHMAKING_TICKETS[ticket_id]["status"] = "expired"
        for queue in MATCHMAKING_QUEUE.values():
            if ticket_id in queue:
                queue.remove(ticket_id)


def _avg_wait_seconds() -> int:
    if RECENT_WAIT_TIMES:
        return int(sum(RECENT_WAIT_TIMES) / len(RECENT_WAIT_TIMES))
    queue_depth = sum(len(queue) for queue in MATCHMAKING_QUEUE.values())
    return max(20, 10 * queue_depth + 15)


def _current_global_status() -> MatchmakingStatus:
    queue_depth = sum(len(queue) for queue in MATCHMAKING_QUEUE.values())
    avg_wait = _avg_wait_seconds()
    issues = list(GLOBAL_MATCHMAKING_STATE.get("issues", []))
    if queue_depth >= 5:
        issues = issues + ["High traffic detected. Expect longer wait times."]
    return MatchmakingStatus(
        avg_wait_seconds=avg_wait,
        active_sessions=GLOBAL_MATCHMAKING_STATE["active_sessions"],
        queue_depth=queue_depth,
        server_status=GLOBAL_MATCHMAKING_STATE["server_status"],
        last_updated=time.time(),
        issues=issues,
    )


def _game_queue_statuses() -> list[GameQueueStatus]:
    statuses: list[GameQueueStatus] = []
    telemetry = conn.hgetall(TELEMETRY_KEY)
    for slug, meta in AVAILABLE_GAMES.items():
        queue_depth = len(MATCHMAKING_QUEUE.get(slug, []))
        recent_matches = [
            rec
            for rec in MATCHMAKING_ASSIGNMENTS
            if rec["game"] == slug and rec.get("matched_at")
        ]
        last_match = recent_matches[-1]["matched_at"] if recent_matches else None
        running_sessions = sum(
            1
            for rec in MATCHMAKING_ASSIGNMENTS
            if rec["game"] == slug and rec["matched_at"] > time.time() - 900
        )
        statuses.append(
            GameQueueStatus(
                slug=slug,
                title=meta["title"],
                queue_depth=queue_depth,
                avg_wait_seconds=_avg_wait_seconds(),
                running_sessions=running_sessions,
                last_match=last_match,
                games_played=int(telemetry.get(f"games_played:{slug}", 0)),
                avg_session_seconds=_compute_avg_session_seconds(slug, telemetry),
                enabled=_is_game_enabled(slug),
            )
        )
    return statuses


def _compute_avg_session_seconds(
    slug: str, telemetry: dict[str, str]
) -> Optional[float]:
    total = float(telemetry.get(f"total_session_seconds:{slug}", 0) or 0)
    count = float(telemetry.get(f"session_count:{slug}", 0) or 0)
    if not count:
        return None
    return total / count


def _record_match_log(entry: MatchLogEntry) -> None:
    conn.lpush(MATCH_LOG_KEY, entry.model_dump_json())
    conn.ltrim(MATCH_LOG_KEY, 0, MAX_MATCH_LOGS - 1)


def _load_match_logs(limit: int = 200) -> list[MatchLogEntry]:
    raw_entries = conn.lrange(MATCH_LOG_KEY, 0, limit - 1)
    entries: list[MatchLogEntry] = []
    for raw in raw_entries:
        try:
            data = json.loads(raw)
            entries.append(MatchLogEntry(**data))
        except Exception:
            continue
    return entries


def _compute_leaderboard(entries: list[MatchLogEntry]) -> list[LeaderboardEntry]:
    aggregated: dict[str, dict[str, float]] = {}
    for entry in entries:
        bucket = aggregated.setdefault(
            entry.game,
            {
                "total": 0,
                "human_wins": 0,
                "ai_wins": 0,
                "duration_sum": 0.0,
            },
        )
        bucket["total"] += 1
        bucket["duration_sum"] += entry.duration_seconds
        if entry.winner == "human":
            bucket["human_wins"] += 1
        else:
            bucket["ai_wins"] += 1
    leaderboard: list[LeaderboardEntry] = []
    for slug, stats in aggregated.items():
        total = max(stats["total"], 1)
        leaderboard.append(
            LeaderboardEntry(
                game=slug,
                total_matches=int(total),
                human_wins=int(stats["human_wins"]),
                ai_wins=int(stats["ai_wins"]),
                human_win_rate=stats["human_wins"] / total,
                avg_duration_seconds=stats["duration_sum"] / total,
            )
        )
    return leaderboard


def _persist_leaderboard(entries: list[LeaderboardEntry]) -> dict[str, typing.Any]:
    payload = {
        "entries": [entry.model_dump() for entry in entries],
        "last_updated": time.time(),
    }
    conn.set(LEADERBOARD_KEY, json.dumps(payload))
    return payload


def _load_leaderboard_cache() -> LeaderboardResponse | None:
    raw = conn.get(LEADERBOARD_KEY)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return LeaderboardResponse(
            entries=[LeaderboardEntry(**entry) for entry in data.get("entries", [])],
            last_updated=data.get("last_updated", time.time()),
        )
    except Exception:
        return None


def _ticket_key(ticket_id: str) -> str:
    return f"{TICKET_KEY_PREFIX}{ticket_id}"


def _persist_queue_snapshot() -> None:
    snapshot = {slug: json.dumps(queue) for slug, queue in MATCHMAKING_QUEUE.items()}
    if snapshot:
        conn.hset(QUEUE_SNAPSHOT_KEY, mapping=snapshot)


def _record_queue_depth(slug: str) -> None:
    conn.hset(
        TELEMETRY_KEY,
        mapping={
            f"queue_depth:{slug}": len(MATCHMAKING_QUEUE.get(slug, [])),
            "last_updated": time.time(),
        },
    )


def _store_ticket(record: dict[str, typing.Any]) -> None:
    conn.hset(
        _ticket_key(record["ticket_id"]),
        mapping={
            "participant_id": record["participant_id"],
            "games": json.dumps(record["games"]),
            "status": record["status"],
            "queued_at": record["queued_at"],
            "matched_at": record.get("matched_at") or "",
            "matched_game": record.get("matched_game") or "",
        },
    )


def _update_ticket(ticket_id: str, **fields: typing.Any) -> None:
    if not fields:
        return
    mapping = {}
    for key, value in fields.items():
        if key == "games":
            mapping[key] = json.dumps(value)
        else:
            mapping[key] = value or ""
    conn.hset(_ticket_key(ticket_id), mapping=mapping)


def _load_ticket(ticket_id: str) -> dict[str, typing.Any] | None:
    record = conn.hgetall(_ticket_key(ticket_id))
    if not record:
        return None
    data: dict[str, typing.Any] = {
        "ticket_id": ticket_id,
        "participant_id": record.get("participant_id"),
        "status": record.get("status") or "queued",
        "queued_at": float(record.get("queued_at") or 0),
        "matched_at": float(record.get("matched_at") or 0) or None,
        "matched_game": record.get("matched_game") or None,
    }
    games_raw = record.get("games")
    if games_raw:
        try:
            data["games"] = json.loads(games_raw)
        except json.JSONDecodeError:
            data["games"] = []
    else:
        data["games"] = []
    return data


def _identity_token_key(token: str) -> str:
    return f"{IDENTITY_TOKEN_KEY}{token}"


def _create_identity(
    participant_id: str, display_name: Optional[str]
) -> IdentityResponse:
    token = str(uuid.uuid4())
    record = {
        "participant_id": participant_id,
        "display_name": display_name or participant_id,
        "created_at": time.time(),
    }
    conn.hset(_identity_token_key(token), mapping=record)
    conn.hset(IDENTITY_PARTICIPANT_KEY, participant_id, token)
    return IdentityResponse(
        token=token,
        participant_id=participant_id,
        display_name=record["display_name"],
        created_at=record["created_at"],
    )


def _get_identity(token: str) -> IdentityResponse | None:
    record = conn.hgetall(_identity_token_key(token))
    if not record:
        return None
    return IdentityResponse(
        token=token,
        participant_id=record.get("participant_id", "unknown"),
        display_name=record.get("display_name"),
        created_at=float(record.get("created_at", 0) or 0),
    )


def _memory_key(participant_id: str) -> str:
    return f"{MEMORY_KEY_PREFIX}{participant_id}".lower()


def _get_memory(participant_id: str) -> MemoryPayload:
    raw = conn.hgetall(_memory_key(participant_id))
    if not raw:
        return MemoryPayload(participant_id=participant_id)
    return MemoryPayload(
        participant_id=participant_id,
        notes=raw.get("notes"),
        role_history=json.loads(raw.get("role_history", "[]")),
        vote_history=json.loads(raw.get("vote_history", "[]")),
        custom=json.loads(raw.get("custom", "{}")),
    )


def _update_memory_record(participant_id: str, payload: MemoryPayload) -> MemoryPayload:
    current = _get_memory(participant_id)
    merged = MemoryPayload(
        participant_id=participant_id,
        notes=payload.notes or current.notes,
        role_history=payload.role_history or current.role_history,
        vote_history=payload.vote_history or current.vote_history,
        custom=payload.custom or current.custom,
    )
    conn.hset(
        _memory_key(participant_id),
        mapping={
            "notes": merged.notes or "",
            "role_history": json.dumps(merged.role_history),
            "vote_history": json.dumps(merged.vote_history),
            "custom": json.dumps(merged.custom),
        },
    )
    return merged


def _is_game_enabled(slug: str) -> bool:
    flag = conn.hget(GAME_STATE_KEY, slug)
    if flag is None:
        return True
    return flag == "enabled"


def _set_game_enabled(slug: str, enabled: bool) -> None:
    conn.hset(GAME_STATE_KEY, slug, "enabled" if enabled else "disabled")


def _require_admin(token: str | None) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin API key not configured.")
    if token != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin token.")


async def _matchmaking_scheduler() -> None:
    while True:
        await asyncio.sleep(3)
        now = time.time()
        async with QUEUE_LOCK:
            _cleanup_expired_tickets(now)
            for slug in AVAILABLE_GAMES:
                queue = MATCHMAKING_QUEUE.get(slug, [])
                if not queue:
                    continue
                ticket_id = queue.pop(0)
                ticket = MATCHMAKING_TICKETS.get(ticket_id)
                if not ticket or ticket["status"] != "queued":
                    continue
                ticket["status"] = "matched"
                ticket["matched_game"] = slug
                ticket["matched_at"] = now
                RECENT_WAIT_TIMES.append(now - ticket["queued_at"])
                _update_ticket(
                    ticket_id,
                    status="matched",
                    matched_game=slug,
                    matched_at=now,
                )
                conn.hincrby(
                    TELEMETRY_KEY,
                    f"games_played:{slug}",
                    1,
                )
                conn.hincrbyfloat(
                    TELEMETRY_KEY,
                    f"total_wait_seconds:{slug}",
                    now - ticket["queued_at"],
                )
                conn.hincrbyfloat(
                    TELEMETRY_KEY,
                    f"total_session_seconds:{slug}",
                    AVAILABLE_GAMES.get(slug, {}).get("est_duration", 600),
                )
                conn.hincrby(
                    TELEMETRY_KEY,
                    f"session_count:{slug}",
                    1,
                )
                _record_queue_depth(slug)
                winner = random.choice(["human", "ai"])
                duration = int(
                    AVAILABLE_GAMES.get(slug, {}).get("est_duration", 600)
                    * random.uniform(0.75, 1.25)
                )
                match_entry = MatchLogEntry(
                    game=slug,
                    ticket_id=ticket_id,
                    participant_id=ticket["participant_id"],
                    opponent_model=f"LLM-{random.randint(1, 4)}",
                    winner=winner,
                    duration_seconds=duration,
                    recorded_at=now,
                )
                _record_match_log(match_entry)
                leaderboard_entries = _compute_leaderboard(_load_match_logs(limit=200))
                _persist_leaderboard(leaderboard_entries)
                MATCHMAKING_ASSIGNMENTS.append(
                    {
                        "ticket_id": ticket_id,
                        "game": slug,
                        "matched_at": now,
                        "participant_id": ticket["participant_id"],
                    }
                )
                GLOBAL_MATCHMAKING_STATE["active_sessions"] = min(
                    GLOBAL_MATCHMAKING_STATE["active_sessions"] + 1, 24
                )
            GLOBAL_MATCHMAKING_STATE["active_sessions"] = max(
                1, GLOBAL_MATCHMAKING_STATE["active_sessions"] - 1
            )
            GLOBAL_MATCHMAKING_STATE["last_updated"] = now
            _persist_queue_snapshot()


@app.on_event("startup")
async def _start_scheduler() -> None:
    global SCHEDULER_TASK
    if SCHEDULER_TASK is None:
        SCHEDULER_TASK = asyncio.create_task(_matchmaking_scheduler())


@app.on_event("shutdown")
async def _stop_scheduler() -> None:
    global SCHEDULER_TASK
    if SCHEDULER_TASK:
        SCHEDULER_TASK.cancel()


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


class MatchmakingQueueRequest(BaseModel):
    participant_id: str
    games: list[str]
    priority: Optional[str] = None


class MatchmakingQueueResponse(CamelModel):
    ticket_id: str
    position: int
    estimated_wait_seconds: int
    status: str
    message: str


class MatchmakingStatus(CamelModel):
    avg_wait_seconds: int
    active_sessions: int
    queue_depth: int
    server_status: str
    last_updated: float
    issues: list[str]


class GameQueueStatus(CamelModel):
    slug: str
    title: str
    queue_depth: int
    avg_wait_seconds: int
    running_sessions: int
    last_match: Optional[float] = None
    games_played: int = 0
    avg_session_seconds: Optional[float] = None
    enabled: bool = True


class QueueOverview(CamelModel):
    global_stats: MatchmakingStatus
    games: list[GameQueueStatus]


class TicketStatus(CamelModel):
    ticket_id: str
    participant_id: str
    games: list[str]
    status: str
    queued_at: float
    matched_at: Optional[float] = None
    matched_game: Optional[str] = None


class MatchLogEntry(CamelModel):
    game: str
    ticket_id: str
    participant_id: str
    opponent_model: str
    winner: str
    duration_seconds: int
    recorded_at: float


class LeaderboardEntry(CamelModel):
    game: str
    total_matches: int
    human_wins: int
    ai_wins: int
    human_win_rate: float
    avg_duration_seconds: float


class LeaderboardResponse(CamelModel):
    entries: list[LeaderboardEntry]
    last_updated: float


class PersonalHistoryEntry(CamelModel):
    game: str
    opponent_model: str
    winner: str
    duration_seconds: int
    recorded_at: float


class PersonalHistoryResponse(CamelModel):
    participant_id: str
    history: list[PersonalHistoryEntry]


class IdentityRequest(BaseModel):
    participant_id: str
    display_name: Optional[str] = None


class IdentityResponse(CamelModel):
    token: str
    participant_id: str
    display_name: Optional[str] = None
    created_at: float


class MemoryPayload(CamelModel):
    participant_id: str
    notes: Optional[str] = None
    role_history: list[str] = Field(default_factory=list)
    vote_history: list[str] = Field(default_factory=list)
    custom: dict[str, typing.Any] = Field(default_factory=dict)


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
    env.setdefault(
        "REDIS_OM_URL", os.environ.get("REDIS_OM_URL", "redis://localhost:6379")
    )
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
async def create_werewolf_game(request: CreateWerewolfGameRequest = Body(...)) -> dict:
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
            detail=f"werewolf_server.py not found at {WEREWOLF_SERVER_PATH}",
        )

    env = os.environ.copy()
    env.setdefault(
        "REDIS_OM_URL", os.environ.get("REDIS_OM_URL", "redis://localhost:6379")
    )

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
        "message": "Game server launching. Poll /games/werewolf/sessions/{session_id} for state.",
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
    session_id: str, participant_id: str, action: WerewolfActionRequest = Body(...)
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

    return {"status": "submitted", "message": "Action queued for processing."}


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


@app.post("/games/queue", response_model=MatchmakingQueueResponse)
async def enqueue_matchmaking(
    request: MatchmakingQueueRequest,
) -> MatchmakingQueueResponse:
    if not request.games:
        raise HTTPException(
            status_code=400, detail="At least one game must be selected."
        )
    valid_games = [
        slug
        for slug in request.games
        if slug in AVAILABLE_GAMES and _is_game_enabled(slug)
    ]
    if not valid_games:
        raise HTTPException(status_code=400, detail="No valid games were selected.")
    ticket_id = str(uuid.uuid4())
    now = time.time()
    async with QUEUE_LOCK:
        record = {
            "ticket_id": ticket_id,
            "participant_id": request.participant_id or "guest",
            "games": valid_games,
            "status": "queued",
            "queued_at": now,
            "matched_at": None,
            "matched_game": None,
        }
        MATCHMAKING_TICKETS[ticket_id] = record
        _store_ticket(record)
        for slug in valid_games:
            MATCHMAKING_QUEUE.setdefault(slug, []).append(ticket_id)
            _record_queue_depth(slug)
        estimated_wait = max(
            _avg_wait_seconds(),
            15 + sum(len(MATCHMAKING_QUEUE[g]) for g in valid_games),
        )
        position = min(len(MATCHMAKING_QUEUE[valid_games[0]]), 99)
    _persist_queue_snapshot()
    message = (
        f"Ticket {ticket_id[:8]} queued for {len(valid_games)} "
        f"game{'s' if len(valid_games) > 1 else ''}. We'll notify you when a slot opens."
    )
    return MatchmakingQueueResponse(
        ticket_id=ticket_id,
        position=position,
        estimated_wait_seconds=estimated_wait,
        status="queued",
        message=message,
    )


@app.get("/games/queue", response_model=QueueOverview)
async def queue_overview() -> QueueOverview:
    async with QUEUE_LOCK:
        _cleanup_expired_tickets()
        return QueueOverview(
            global_stats=_current_global_status(),
            games=_game_queue_statuses(),
        )


@app.get("/games/queue/tickets/{ticket_id}", response_model=TicketStatus)
async def ticket_status(ticket_id: str) -> TicketStatus:
    async with QUEUE_LOCK:
        ticket = MATCHMAKING_TICKETS.get(ticket_id) or _load_ticket(ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found.")
        return TicketStatus(
            ticket_id=ticket_id,
            participant_id=ticket["participant_id"],
            games=ticket["games"],
            status=ticket["status"],
            queued_at=ticket["queued_at"],
            matched_at=ticket.get("matched_at"),
            matched_game=ticket.get("matched_game"),
        )


@app.delete("/games/queue/tickets/{ticket_id}")
async def cancel_ticket(ticket_id: str) -> dict:
    async with QUEUE_LOCK:
        ticket = MATCHMAKING_TICKETS.get(ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found.")
        if ticket["status"] == "queued":
            for slug in ticket["games"]:
                queue = MATCHMAKING_QUEUE.get(slug, [])
                if ticket_id in queue:
                    queue.remove(ticket_id)
                    _record_queue_depth(slug)
            ticket["status"] = "cancelled"
            _update_ticket(ticket_id, status="cancelled")
            _persist_queue_snapshot()
        return {"status": ticket["status"]}


@app.get("/games/matchmaking/status", response_model=MatchmakingStatus)
async def matchmaking_status() -> MatchmakingStatus:
    async with QUEUE_LOCK:
        return _current_global_status()


@app.post("/games/matchmaking/queue", response_model=MatchmakingQueueResponse)
async def legacy_enqueue_matchmaking(
    request: MatchmakingQueueRequest,
) -> MatchmakingQueueResponse:
    return await enqueue_matchmaking(request)


@app.post("/games/leaderboard/logs")
async def create_match_log(entry: MatchLogEntry) -> dict:
    async with QUEUE_LOCK:
        _record_match_log(entry)
        conn.hincrby(TELEMETRY_KEY, f"games_played:{entry.game}", 1)
        conn.hincrbyfloat(
            TELEMETRY_KEY,
            f"total_session_seconds:{entry.game}",
            entry.duration_seconds,
        )
        conn.hincrby(TELEMETRY_KEY, f"session_count:{entry.game}", 1)
        if entry.winner == "human":
            conn.hincrby(TELEMETRY_KEY, f"human_wins:{entry.game}", 1)
        else:
            conn.hincrby(TELEMETRY_KEY, f"ai_wins:{entry.game}", 1)
        leaderboard = _compute_leaderboard(_load_match_logs(limit=200))
        payload = _persist_leaderboard(leaderboard)
    return {"status": "recorded", "last_updated": payload["last_updated"]}


@app.get("/games/leaderboard", response_model=LeaderboardResponse)
async def leaderboard() -> LeaderboardResponse:
    cache = _load_leaderboard_cache()
    if cache:
        return cache
    entries = _compute_leaderboard(_load_match_logs(limit=200))
    payload = _persist_leaderboard(entries)
    return LeaderboardResponse(
        entries=entries,
        last_updated=payload["last_updated"],
    )


@app.get("/games/history/{participant_id}", response_model=PersonalHistoryResponse)
async def personal_history(
    participant_id: str,
    limit: int = 20,
) -> PersonalHistoryResponse:
    logs = _load_match_logs(limit=200)
    history = [
        entry
        for entry in logs
        if entry.participant_id.lower() == participant_id.lower()
    ][:limit]
    return PersonalHistoryResponse(
        participant_id=participant_id,
        history=history,
    )


@app.post("/auth/identity", response_model=IdentityResponse)
async def create_identity(request: IdentityRequest) -> IdentityResponse:
    participant_id = request.participant_id.strip()
    if not participant_id:
        raise HTTPException(status_code=400, detail="Participant ID required")
    return _create_identity(participant_id, request.display_name)


@app.get("/auth/identity/{token}", response_model=IdentityResponse)
async def read_identity(token: str) -> IdentityResponse:
    identity = _get_identity(token)
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found")
    return identity


@app.get("/memory/{participant_id}", response_model=MemoryPayload)
async def read_memory(participant_id: str) -> MemoryPayload:
    return _get_memory(participant_id)


@app.post("/memory/{participant_id}", response_model=MemoryPayload)
async def update_memory(
    participant_id: str,
    payload: MemoryPayload,
) -> MemoryPayload:
    return _update_memory_record(participant_id, payload)


def _redis_alive() -> bool:
    try:
        conn.ping()
        return True
    except Exception:
        return False


class AdminGameToggle(BaseModel):
    enabled: bool


class AdminStatusResponse(CamelModel):
    uptime_seconds: float
    redis_alive: bool
    global_stats: MatchmakingStatus
    games: list[GameQueueStatus]


@app.get("/admin/status", response_model=AdminStatusResponse)
async def admin_status(x_admin_token: str = Header(None)) -> AdminStatusResponse:
    _require_admin(x_admin_token)
    async with QUEUE_LOCK:
        global_stats = _current_global_status()
        games = _game_queue_statuses()
    return AdminStatusResponse(
        uptime_seconds=time.time() - START_TIME,
        redis_alive=_redis_alive(),
        global_stats=global_stats,
        games=games,
    )


@app.post("/admin/games/{slug}")
async def admin_toggle_game(
    slug: str,
    payload: AdminGameToggle,
    x_admin_token: str = Header(None),
) -> dict:
    _require_admin(x_admin_token)
    if slug not in AVAILABLE_GAMES:
        raise HTTPException(status_code=404, detail="Unknown game.")
    _set_game_enabled(slug, payload.enabled)
    return {"slug": slug, "enabled": payload.enabled}


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
