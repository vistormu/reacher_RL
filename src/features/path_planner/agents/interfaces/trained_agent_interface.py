import numpy as np

from abc import ABC, abstractmethod


class ITrainedAgent(ABC):
    @abstractmethod
    def get_action(self, observation: np.ndarray) -> int:
        pass
