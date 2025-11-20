from pydantic import BaseModel
from redis_om import JsonModel
from redis_om.model.model import Field

from .base_models import patch_model_for_local_storage
from .storage_backend import is_local_backend


class BaseEnvAgentComboStorage(BaseModel):
    env_id: str = Field(default_factory=lambda: "", index=True)
    agent_ids: list[str] = Field(default_factory=lambda: [], index=True)


if is_local_backend():

    class EnvAgentComboStorage(BaseEnvAgentComboStorage):  # type: ignore[no-redef]
        pass
else:

    class EnvAgentComboStorage(BaseEnvAgentComboStorage, JsonModel):
        pass


# Patch model class for local storage support
EnvAgentComboStorage = patch_model_for_local_storage(EnvAgentComboStorage)
