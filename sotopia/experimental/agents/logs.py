from redis_om import JsonModel
from sotopia.database.logs import BaseEpisodeLog
from sotopia.database.persistent_profile import AgentProfile


class EpisodeLog(BaseEpisodeLog, JsonModel):
    def render_for_humans(self) -> tuple[list[AgentProfile], list[str]]:
        raise NotImplementedError
