from pydantic import BaseModel
from redis_om.model.model import Field
from redis_om import JsonModel

from .base_models import patch_model_for_local_storage
from .storage_backend import is_local_backend


class BaseAnnotator(BaseModel):
    name: str = Field(index=True)
    email: str = Field(index=True)


if is_local_backend():

    class Annotator(BaseAnnotator):  # type: ignore[no-redef]
        pass
else:

    class Annotator(BaseAnnotator, JsonModel):
        pass


# Patch model class for local storage support
Annotator = patch_model_for_local_storage(Annotator)
