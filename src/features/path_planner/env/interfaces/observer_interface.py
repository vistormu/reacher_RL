import numpy as np
from abc import ABC, abstractmethod


class IObserver(ABC):
    def __init__(self) -> None:
        self.low: np.ndarray
        self.high: np.ndarray

        self.observation_space_size: int

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_reward(self) -> float:
        pass

    @abstractmethod
    def is_truncated(self) -> bool:
        pass

    @abstractmethod
    def is_terminated(self) -> bool:
        pass

    @abstractmethod
    def get_info(self) -> dict:
        pass
