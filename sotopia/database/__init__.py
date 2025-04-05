from typing import TypeVar
from logging import Logger
from rich import print as rprint
import redis
import os
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
    BaseCustomEvaluationDimension,
    CustomEvaluationDimensionList,
    BaseCustomEvaluationDimensionList,
    GoalDimension,
    SotopiaDimensions,
    SotopiaDimensionsPlus,
)

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
    "BaseCustomEvaluationDimension",
    "CustomEvaluationDimensionList",
    "BaseCustomEvaluationDimensionList",
    "NonStreamingSimulationStatus",
    "GoalDimension",
    "SotopiaDimensions",
    "SotopiaDimensionsPlus",
]

InheritedJsonModel = TypeVar("InheritedJsonModel", bound="JsonModel")


def _json_model_all(cls: type[InheritedJsonModel]) -> list[InheritedJsonModel]:
    return cls.find().all()  # type: ignore[return-value]


JsonModel.all = classmethod(_json_model_all)  # type: ignore[assignment,method-assign]
logger = Logger("sotopia.database")

# Test Redis connection before proceeding with any database operations
try:
    redis_url = os.getenv("REDIS_OM_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url)
    redis_client.ping()
    rprint(f"[green]Successfully connected to Redis database {redis_url}[/green]")
except redis.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {e}")
    rprint(f"[red]Failed to connect to Redis database {redis_url}[/red]")
try:
    Migrator().run()
except Exception as e:
    logger.debug(
        f"Error running migrations: {e} This is expected if you have not set up redis yet."
    )

# Try Redis OM connection
try:
    # Initialize an empty JsonModel to ensure model is registered
    JsonModel()
    rprint("[green]Successfully initialized Redis OM object[/green].")
except Exception as e:
    logger.error(
        f"Failed to initialize Redis OM object: {e}. The connection to your redis database might be problematic."
    )
    rprint("[red]Failed to initialize Redis OM object[/red]")
