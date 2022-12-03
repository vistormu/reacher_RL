import numpy as np

from .....core.entities import Point, Vector
from .use_cases import Observer, Translator


class GridWorldEnvToDeploy:
    def __init__(self, size: float) -> None:
        # Env variables
        self.size: float = size
        self.limits: list[tuple[float, float, float]] = [(0.0, 0.6),
                                                         (0.0, 0.6),
                                                         (0.7, 1.5)]

        # Entities
        self.target: Point = None
        self.virtual_point: Point = None

        # Observer and translator
        self.observer: Observer = Observer(self.size, self.limits)
        self.translator: Translator = Translator()

    def init(self, virtual_point: Point, target: Point, return_info: bool = False):
        # Initialize entities
        self.target = target
        self.virtual_point = virtual_point

        observation: np.ndarray = self.observer.get_observation(self.virtual_point,
                                                                self.target)
        info: dict = self.observer.get_info(self.virtual_point,
                                            self.target)

        return (observation, info) if return_info else observation

    def step(self, action: int) -> tuple[np.ndarray, bool, dict]:
        # Move the virtual point from the action
        direction: Vector = self.translator.get_direction(action, self.size)
        new_position: Point = self.virtual_point + Point(*direction)

        self.virtual_point = Point(np.clip(new_position.x, *self.limits[0]),
                                   np.clip(new_position.y, *self.limits[1]),
                                   np.clip(new_position.z, *self.limits[2]))

        # Get observation, reward, done and info
        observation: np.ndarray = self.observer.get_observation(self.virtual_point,
                                                                self.target)
        done: bool = self.observer.is_done(self.virtual_point,
                                           self.target)
        info: dict = self.observer.get_info(self.virtual_point, self.target)

        return observation, done, info