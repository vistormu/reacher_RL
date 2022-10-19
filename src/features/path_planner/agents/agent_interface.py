from abc import ABC, abstractmethod
import numpy as np


class IAgent(ABC):

    @abstractmethod
    def get_action(self, observation: np.ndarray):
        pass

    @abstractmethod
    def train(self, observation: np.ndarray, new_observation: np.ndarray, action, reward: int, done: bool) -> None:
        pass

    @abstractmethod
    def decay(self) -> None:
        pass

    @abstractmethod
    def save_model(self) -> None:
        pass

    @abstractmethod
    def load_model(self) -> None:
        pass
