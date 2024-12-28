from typing import TypeVar
from redis_om import JsonModel, Migrator
from .annotators import Annotator
from .env_agent_combo_storage import EnvAgentComboStorage
from .logs import (
    AnnotationForEpisode,
    BaseEpisodeLog,
    NonStreamingSimulationStatus,
    EpisodeLog,
)
from .persistent_profile import (
    AgentProfile,
    BaseAgentProfile,
    EnvironmentProfile,
    BaseEnvironmentProfile,
    BaseRelationshipProfile,
    RelationshipProfile,
    RelationshipType,
)
from .serialization import (
    agentprofiles_to_csv,
    agentprofiles_to_jsonl,
    envagnetcombostorage_to_csv,
    envagnetcombostorage_to_jsonl,
    environmentprofiles_to_csv,
    environmentprofiles_to_jsonl,
    episodes_to_csv,
    episodes_to_jsonl,
    get_rewards_from_episode,
    jsonl_to_agentprofiles,
    jsonl_to_envagnetcombostorage,
    jsonl_to_environmentprofiles,
    jsonl_to_episodes,
    jsonl_to_relationshipprofiles,
    relationshipprofiles_to_csv,
    relationshipprofiles_to_jsonl,
)
from .session_transaction import MessageTransaction, SessionTransaction
from .waiting_room import MatchingInWaitingRoom
from .aggregate_annotations import map_human_annotations_to_episode_logs
from .evaluation_dimensions import (
    EvaluationDimensionBuilder,
    CustomEvaluationDimension,
    CustomEvaluationDimensionList,
)

from logging import Logger

logger = Logger("sotopia.database")

__all__ = [
    "AgentProfile",
    "BaseAgentProfile",
    "EnvironmentProfile",
    "BaseEnvironmentProfile",
    "EpisodeLog",
    "BaseEpisodeLog",
    "NonStreamingSimulationStatus",
    "EnvAgentComboStorage",
    "AnnotationForEpisode",
    "Annotator",
    "BaseRelationshipProfile",
    "RelationshipProfile",
    "RelationshipType",
    "RedisCommunicationMixin",
    "SessionTransaction",
    "MessageTransaction",
    "MatchingInWaitingRoom",
    "agentprofiles_to_csv",
    "agentprofiles_to_jsonl",
    "environmentprofiles_to_csv",
    "environmentprofiles_to_jsonl",
    "relationshipprofiles_to_csv",
    "relationshipprofiles_to_jsonl",
    "envagnetcombostorage_to_csv",
    "envagnetcombostorage_to_jsonl",
    "episodes_to_csv",
    "episodes_to_jsonl",
    "map_human_annotations_to_episode_logs",
    "jsonl_to_agentprofiles",
    "jsonl_to_environmentprofiles",
    "jsonl_to_episodes",
    "jsonl_to_relationshipprofiles",
    "jsonl_to_envagnetcombostorage",
    "get_rewards_from_episode",
    "EvaluationDimensionBuilder",
    "CustomEvaluationDimension",
    "CustomEvaluationDimensionList",
    "NonStreamingSimulationStatus",
]

InheritedJsonModel = TypeVar("InheritedJsonModel", bound="JsonModel")


def _json_model_all(cls: type[InheritedJsonModel]) -> list[InheritedJsonModel]:
    return cls.find().all()  # type: ignore[return-value]


JsonModel.all = classmethod(_json_model_all)  # type: ignore[assignment,method-assign]

try:
    Migrator().run()
except Exception as e:
    logger.debug(
        f"Error running migrations: {e} This is expected if you have not set up redis yet."
    )
