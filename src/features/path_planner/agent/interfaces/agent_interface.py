import numpy as np

from abc import ABC, abstractmethod


class IAgent(ABC):

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> int:
        pass

    @abstractmethod
    def train(self, observation: np.ndarray, new_observation: np.ndarray, action: int, reward: float, done: bool) -> None:
        pass

    @abstractmethod
    def save_model(self, name: str) -> None:
        pass

    @abstractmethod
    def load_model(self, name: str) -> None:
        pass
