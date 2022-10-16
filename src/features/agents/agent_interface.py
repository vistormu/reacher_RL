from abc import ABC, abstractmethod
import numpy as np


class IAgent(ABC):

    @abstractmethod
    def get_action(self, observation: np.ndarray):
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def decay(self) -> None:
        pass
