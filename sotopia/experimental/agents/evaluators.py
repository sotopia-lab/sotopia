from abc import ABC, abstractmethod
from .logs import EpisodeLog


class BaseEvaluator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, epilog: EpisodeLog) -> tuple[float, str]:
        """
        evaluate an episode, returns the score and reward prompt
        """
        raise NotImplementedError


class DummyEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, epilog: EpisodeLog) -> tuple[float, str]:
        """
        evaluate an episode, returns the score and reward prompt
        """
        return 0.0, "No evaluation implemented"
