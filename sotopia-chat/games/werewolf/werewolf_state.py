"""Utilities for reading and writing Werewolf game state to Redis."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import redis.asyncio as redis

STATE_PREFIX = "werewolf:session"


def state_key(session_id: str) -> str:
    return f"{STATE_PREFIX}:{session_id}:state"


def action_key(session_id: str, participant_id: str) -> str:
    return f"{STATE_PREFIX}:{session_id}:action:{participant_id}"


class WerewolfStateStore:
    """Async wrapper around Redis for Werewolf state persistence."""

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url

    def _client(self, *, decode: bool = True) -> redis.Redis:
        return redis.Redis.from_url(
            self.redis_url,
            decode_responses=decode,
        )

    async def write_state(
        self,
        session_id: str,
        payload: Dict[str, Any],
        *,
        ttl: int | None = None,
    ) -> None:
        client = self._client()
        try:
            await client.set(
                state_key(session_id),
                json.dumps(payload),
                ex=ttl,
            )
        finally:
            await client.aclose()

    async def read_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        client = self._client()
        try:
            raw = await client.get(state_key(session_id))
        finally:
            await client.aclose()

        if raw is None:
            return None
        return json.loads(raw)

    async def delete_state(self, session_id: str) -> None:
        client = self._client()
        try:
            await client.delete(state_key(session_id))
        finally:
            await client.aclose()

    async def push_action(
        self,
        session_id: str,
        participant_id: str,
        action_type: str,
        argument: str,
        *,
        ttl: int = 60,
    ) -> None:
        client = self._client()
        try:
            payload = {
                "action_type": action_type,
                "argument": argument,
                "timestamp": time.time(),
            }
            await client.set(
                action_key(session_id, participant_id),
                json.dumps(payload),
                ex=ttl,
            )
        finally:
            await client.aclose()

    async def pop_action(
        self,
        session_id: str,
        participant_id: str,
    ) -> Optional[Dict[str, Any]]:
        client = self._client()
        key = action_key(session_id, participant_id)
        try:
            pipe = client.pipeline()
            pipe.get(key)
            pipe.delete(key)
            data, _ = await pipe.execute()
        finally:
            await client.aclose()

        if data is None:
            return None
        return json.loads(data)

    async def delete_action(
        self,
        session_id: str,
        participant_id: str,
    ) -> None:
        client = self._client()
        try:
            await client.delete(action_key(session_id, participant_id))
        finally:
            await client.aclose()
