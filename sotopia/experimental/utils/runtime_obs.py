from enum import Enum
from pydantic import Field

from aact.messages import DataModel
from aact.messages.registry import DataModelFactory

class ObservationType(Enum):
    BROWSER_OUTPUT = "BrowserOutputObservation"
    CMD_OUTPUT = "CmdOutputObservation"
    FILE_WRITE = "FileWriteObservation"
    FILE_READ = "FileReadObservation"
    ERROR = "ErrorObservation"
    

@DataModelFactory.register("runtime_observation")
class RuntimeObservation(DataModel):
    observation_type: ObservationType = Field(
        description="The type of runtime observation"
    )
    details: str = Field(
        description="Details about the observation"
    )

    def to_natural_language(self) -> str:
        observation_descriptions = {
            ObservationType.BROWSER_OUTPUT: f"observed browser output: {self.details}",
            ObservationType.CMD_OUTPUT: f"observed command output: {self.details}",
            ObservationType.FILE_WRITE: f"observed file write: {self.details}",
            ObservationType.FILE_READ: f"observed file read: {self.details}",
            ObservationType.ERROR: f"observed an error: {self.details}",
        }

        return observation_descriptions.get(self.observation_type, "observed an unknown event")