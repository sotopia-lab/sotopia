from typing import TYPE_CHECKING
from pydantic import BaseModel
from redis_om import JsonModel
from redis_om.model.model import Field

from .auto_expires_mixin import AutoExpireMixin
from .base_models import patch_model_for_local_storage
from .storage_backend import is_local_backend


class BaseMatchingInWaitingRoom(BaseModel):
    pk: str | None = Field(default=None)
    timestamp: float = Field()
    client_ids: list[str] = Field(default_factory=list)
    session_ids: list[str] = Field(default_factory=list)
    session_id_retrieved: list[str] = Field(default_factory=list)


if TYPE_CHECKING:
    # For type checking, always assume Redis backend to get proper method signatures
    class MatchingInWaitingRoom(AutoExpireMixin, BaseMatchingInWaitingRoom, JsonModel):
        pass
elif is_local_backend():
    # For local backend, inherit only from BaseMatchingInWaitingRoom (no TTL support)
    class MatchingInWaitingRoom(BaseMatchingInWaitingRoom):
        pass
else:
    # For Redis backend, inherit from AutoExpireMixin and JsonModel
    class MatchingInWaitingRoom(AutoExpireMixin, BaseMatchingInWaitingRoom, JsonModel):  # type: ignore[no-redef]
        pass


# Patch model class for local storage support
# Note: TTL/expiration is not supported in local storage mode
MatchingInWaitingRoom = patch_model_for_local_storage(MatchingInWaitingRoom)  # type: ignore[misc]
