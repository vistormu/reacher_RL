import numpy as np

from abc import ABC, abstractmethod


class IEnv(ABC):
    def __init__(self) -> None:
        self.action_space_size: int
        self.observation_space_size: int

    @abstractmethod
    def reset(self) -> tuple[np.ndarray, dict]:
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, int, bool, dict]:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
