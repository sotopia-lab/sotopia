from typing import TYPE_CHECKING

from pydantic import BaseModel, field_validator
from redis_om import EmbeddedJsonModel, JsonModel
from redis_om.model.model import Field

from .auto_expires_mixin import AutoExpireMixin
from .base_models import patch_model_for_local_storage
from .storage_backend import is_local_backend


class MessageTransaction(EmbeddedJsonModel):
    timestamp_str: str = Field(index=True)
    sender: str = Field(index=True)
    message: str

    def to_tuple(self) -> tuple[float, str, str]:
        return (
            float(self.timestamp_str),
            self.sender,
            self.message,
        )


class BaseSessionTransaction(BaseModel):
    session_id: str = Field(index=True)
    client_id: str = Field(index=True)
    server_id: str = Field(index=True)
    client_action_lock: str = Field(default="no action")
    message_list: list[MessageTransaction] = Field(
        description="""List of messages in this session.
    Each message is a tuple of (timestamp, sender_id, message)
    The message list should be sorted by timestamp.
    """
    )

    @field_validator("message_list")
    def validate_message_list(
        cls, v: list[MessageTransaction]
    ) -> list[MessageTransaction]:
        def _is_sorted(x: list[MessageTransaction]) -> bool:
            return all(
                float(x[i].timestamp_str) <= float(x[i + 1].timestamp_str)
                for i in range(len(x) - 1)
            )

        assert _is_sorted(v), "Message list should be sorted by timestamp"
        return v


if TYPE_CHECKING:
    # For type checking, always assume Redis backend to get proper method signatures
    class SessionTransaction(AutoExpireMixin, BaseSessionTransaction, JsonModel):
        pass
elif is_local_backend():
    # For local backend, inherit only from BaseSessionTransaction (no TTL support)
    class SessionTransaction(BaseSessionTransaction):
        pass
else:
    # For Redis backend, inherit from AutoExpireMixin and JsonModel
    class SessionTransaction(AutoExpireMixin, BaseSessionTransaction, JsonModel):  # type: ignore[no-redef]
        pass


# Patch model class for local storage support
# Note: TTL/expiration is not supported in local storage mode
SessionTransaction = patch_model_for_local_storage(SessionTransaction)  # type: ignore[misc]
