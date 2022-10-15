import numpy as np

from ....interfaces import ITranslator
from src.core.entities import Point


class Translator(ITranslator):
    def __init__(self) -> None:
        self._action_to_direction = {
            0: np.array([1, 0, 0]),
            1: np.array([-1, 0, 0]),
            2: np.array([0, 1, 0]),
            3: np.array([0, -1, 0]),
            4: np.array([0, 0, 1]),
            5: np.array([0, 0, -1]),
        }

        self.increment = 0.01
        self.action_space_size = len(self._action_to_direction)

    def get_direction(self, action):
        direction = self._action_to_direction[action] * self.increment

        return Point(*direction)
