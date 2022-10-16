from gym import spaces
import numpy as np

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
        self.target: Point = Point(0.0, 0.0, 0.0)
        self.virtual_point: Point = Point(0.0, 0.0, 0.0)

        # Translator and observer
        self.translator: Translator = Translator()
        self.observer: Observer = Observer(self.size, self.limits)

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

        # Get observation and info
        observation: np.ndarray = self.observer.get_observation(self.virtual_point,
                                                                self.target)
        info: dict = self.observer.get_info(self.virtual_point, self.target)

        return (observation, info) if return_info else observation

    def step(self, action: int) -> tuple[np.ndarray, int, bool, dict]:
        # Move the virtual point from the action
        direction: Point = self.translator.get_direction(action)
        new_position: Point = self.virtual_point + direction

        self.virtual_point = Point(np.clip(new_position.x, self.limits[0][0], self.limits[0][1]),
                                   np.clip(
                                       new_position.y, self.limits[1][0], self.limits[2][1]),
                                   np.clip(new_position.z, self.limits[2][0], self.limits[2][1]),)

        # Get observation, reward, done and info
        observation = self.observer.get_observation(self.virtual_point,
                                                    self.target)
        reward = self.observer.get_reward(self.virtual_point,
                                          self.target)
        done = self.observer.is_done(self.virtual_point,
                                     self.target)
        info = self.observer.get_info(self.virtual_point, self.target)

        return observation, reward, done, info

    def render():
        pass
