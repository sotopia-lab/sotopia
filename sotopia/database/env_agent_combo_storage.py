from typing import TYPE_CHECKING
from pydantic import BaseModel
from redis_om import JsonModel
from redis_om.model.model import Field

from .base_models import patch_model_for_local_storage
from .storage_backend import is_local_backend


class BaseEnvAgentComboStorage(BaseModel):
    pk: str | None = Field(default=None)
    env_id: str = Field(default="", index=True)
    agent_ids: list[str] = Field(default_factory=list, index=True)


if TYPE_CHECKING:
    # For type checking, always assume Redis backend to get proper method signatures
    class EnvAgentComboStorage(BaseEnvAgentComboStorage, JsonModel):
        pass
elif is_local_backend():

    class EnvAgentComboStorage(BaseEnvAgentComboStorage):
        pass
else:

    class EnvAgentComboStorage(BaseEnvAgentComboStorage, JsonModel):  # type: ignore[no-redef]
        pass


# Patch model class for local storage support
EnvAgentComboStorage = patch_model_for_local_storage(EnvAgentComboStorage)  # type: ignore[misc]
