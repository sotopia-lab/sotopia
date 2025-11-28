from typing import TYPE_CHECKING

from pydantic import BaseModel
from redis_om import JsonModel
from redis_om.model.model import Field

from .base_models import patch_model_for_local_storage
from .storage_backend import is_local_backend


class BaseAnnotator(BaseModel):
    pk: str | None = Field(default_factory=lambda: "")
    name: str = Field(index=True)
    email: str = Field(index=True)


if TYPE_CHECKING:
    # For type checking, always assume Redis backend to get proper method signatures
    class Annotator(BaseAnnotator, JsonModel):
        pass
elif is_local_backend():

    class Annotator(BaseAnnotator):
        pass
else:

    class Annotator(BaseAnnotator, JsonModel):  # type: ignore[no-redef]
        pass


# Patch model class for local storage support
Annotator = patch_model_for_local_storage(Annotator)  # type: ignore[misc]
