from ..interfaces import ITranslator
from ..entities import Point


class Translator(ITranslator):
    def __init__(self) -> None:
        self._action_to_direction: dict[int, Point] = {
            0: Point(1, 0, 0),
            1: Point(-1, 0, 0),
            2: Point(0, 1, 0),
            3: Point(0, -1, 0),
            4: Point(0, 0, 1),
            5: Point(0, 0, -1),
        }

        self.action_space_size = len(self._action_to_direction)

    def get_direction(self, action: int) -> Point:
        return self._action_to_direction[action]
