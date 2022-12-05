import numpy as np

from abc import ABC, abstractmethod


class IEnvToDeploy(ABC):
    @abstractmethod
    def init(self) -> tuple[np.ndarray, dict]:
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, bool, dict]:
        pass
