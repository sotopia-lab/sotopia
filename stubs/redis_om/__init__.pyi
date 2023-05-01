import abc
from typing import Any

from pydantic import BaseModel, Field
from pydantic.main import ModelMetaclass

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
    def get(cls, pk: Any) -> "JsonModel": ...
    def save(self) -> None: ...
