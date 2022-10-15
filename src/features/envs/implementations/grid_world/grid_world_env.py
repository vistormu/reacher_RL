from gym import spaces

from ...interfaces import IEnv
from .use_cases import Observer, Translator
from src.core.entities import Point


class GridWorldEnv(IEnv):
    def __init__(self) -> None:
        self.size: float = 0.01
        self.limits: list[tuple[float, float]] = [(0.0, 0.5),
                                                  (0.0, 0.5),
                                                  (1.0, 1.5)]

        # Entities
        target: Point = Point(0.0, 0.0, 0.0)
        virtual_point: Point = Point(0.0, 0.0, 0.0)

        # Translator and observer
        self.translator: Translator = Translator()
        self.observer: Observer = Observer()

        # Action space
        self.action_space_size: int = self.translator.action_space_size
        self.action_space: spaces.Discrete = spaces.Discrete(
            self.action_space_size)

        # Oservation space
        self.observation_space_size: int = self.observer.observation_space_size
        self.observation_space: spaces.Box = spaces.Box(low=self.observer.low,
                                                        high=self.observer.high)

    def reset(self, seed: int = None, return_info: bool = False, options=None):
        super().reset(seed=seed)

        # Set target to a new position
        target_x: float = self.np_random.uniform(*self.limits[0])
        target_y: float = self.np_random.uniform(*self.limits[1])
        target_z: float = self.np_random.uniform(*self.limits[2])

        self.target = Point(target_x, target_y, target_z)

    def step(self):
        pass

    def render():
        pass
