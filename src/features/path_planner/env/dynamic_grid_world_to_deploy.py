import numpy as np
from typing import Optional

from .use_cases import Observer, Translator
from .entities import Point, OccupancyGrid


class DynamicGridworldToDeploy:
    def __init__(self, size: int) -> None:
        # Variables
        self.size: int = size

        # Observer and translator
        self.observer: Observer = Observer(size)
        self.translator: Translator = Translator()

        # Entities
        self.moving_point: Point = Point(0, 0, 0)
        self.target: Point = Point(0, 0, 0)
        self.map: Optional[OccupancyGrid] = None

    def init(self, moving_point: Point, target: Point, map: Optional[OccupancyGrid]) -> tuple[np.ndarray, dict]:
        # Set moving point and target
        self.moving_point = moving_point
        self.target = target

        # Get observation and info
        observation: np.ndarray = self.observer.get_observation(moving_point, target, map)
        info: dict = self.observer.get_info(moving_point, target, map)

        return (observation, info)

    def step(self, action) -> tuple[np.ndarray, bool, bool, dict]:
        # Update moving point
        next_position: Point = self.moving_point + self.translator.get_direction(action)
        self.moving_point = Point(*np.clip(next_position, [0]*3, [self.size-1]*3))

        # Get observation, reward, terminated, truncated and info
        observation: np.ndarray = self.observer.get_observation(self.moving_point, self.target, self.map)
        terminated: bool = self.observer.is_terminated(self.moving_point, self.target, self.map)
        truncated: bool = self.observer.is_truncated(self.moving_point, self.target, self.map)
        info: dict = self.observer.get_info(self.moving_point, self.target, self.map)

        return (observation, terminated, truncated, info)
