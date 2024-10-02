from redis_om.model.model import Field
from redis_om import JsonModel


class Annotator(JsonModel):
    name: str = Field(index=True)
    email: str = Field(index=True)
