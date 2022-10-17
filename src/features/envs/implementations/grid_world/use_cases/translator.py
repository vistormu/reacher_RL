from ....interfaces import ITranslator
from src.core.entities import Vector


class Translator(ITranslator):
    def __init__(self) -> None:
        self._action_to_direction: dict[int, Vector] = {
            0: Vector(1, 0, 0),
            1: Vector(-1, 0, 0),
            2: Vector(0, 1, 0),
            3: Vector(0, -1, 0),
            4: Vector(0, 0, 1),
            5: Vector(0, 0, -1),
        }

        self.action_space_size = len(self._action_to_direction)

    def get_direction(self, action: int, increment: float) -> Vector:
        direction: Vector = self._action_to_direction[action] * increment

        return direction
