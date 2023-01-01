import numpy as np

from typing import Optional
from abc import ABC, abstractmethod


class IEnv(ABC):
    def __init__(self) -> None:
        self.action_space_size: int
        self.observation_space_size: int
        self.reward_range: tuple[float, float]

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        pass

    @abstractmethod
    def render(self) -> None:
        pass
