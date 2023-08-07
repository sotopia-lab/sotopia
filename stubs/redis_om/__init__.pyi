import abc
from typing import Any, Generator, TypeVar

from pydantic import BaseModel, Field
from pydantic.main import ModelMetaclass
from redis_om.model.model import FindQuery

InheritedJsonModel = TypeVar("InheritedJsonModel", bound="JsonModel")

class ModelMeta(ModelMetaclass): ...

class RedisModel(BaseModel, abc.ABC, metaclass=ModelMeta):
    pk: str | None = Field(default=None, primary_key=True)

    @classmethod
    def delete(cls, pk: Any) -> None: ...

class HashModel(RedisModel, abc.ABC):
    @classmethod
    def get(cls, pk: Any) -> "HashModel": ...
    def save(self) -> None: ...

class JsonModel(RedisModel, abc.ABC):
    @classmethod
    def get(cls: type[InheritedJsonModel], pk: Any) -> InheritedJsonModel: ...
    @classmethod
    def all_pks(cls) -> Generator[str, None, None]: ...
    @classmethod
    def find(cls, *args: Any, **kwargs: Any) -> FindQuery: ...
    def save(self) -> None: ...
