import numpy as np

from typing import Optional

from .interfaces import IEnv
from .use_cases import Translator, Observer, Graphics
from .entities import OccupancyGrid, Point

RENDER_FPS: int = 30


class DynamicGridworld(IEnv):
    def __init__(self, size: int, obstacles_enabled: bool = True) -> None:
        # Env variables
        self.size: int = size
        self.obstacles_enabled: bool = obstacles_enabled

        # Use cases
        self.translator: Translator = Translator()
        self.observer: Observer = Observer(size)
        self.graphics: Graphics = Graphics()

        # Entities
        self.map: Optional[OccupancyGrid] = None
        self.target: Point = Point(0, 0, 0)
        self.moving_point: Point = Point(0, 0, 0)

        # Spaces
        self.action_space_size = self.translator.action_space_size
        self.observation_space_size = self.observer.observation_space_size

        # Other variables
        self.window: bool = False
        self.episode_steps: int = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        self.episode_steps = 0

        # Create new map
        if seed is None:
            seed = 1

        if self.obstacles_enabled:
            self.map = OccupancyGrid(self.size, seed)

        # Place target
        if self.map is not None:
            x, y = self.map.get_max_index()
            z = self.map.get_value(x, y) + 1
        else:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            z = np.random.randint(0, self.size)

        self.target = Point(x, y, z)

        # Place moving point
        if self.map is not None:
            x, y = self.map.get_min_index()
            z = self.map.get_value(x, y) + 1
        else:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            z = np.random.randint(0, self.size)

        self.moving_point = Point(x, y, z)

        # Get observation and info
        observation: np.ndarray = self.observer.get_observation(self.moving_point, self.target, self.map)
        info: dict = self.observer.get_info(self.moving_point, self.target, self.map)

        return (observation, info)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.episode_steps += 1

        # Update moving point
        next_position: Point = self.moving_point + self.translator.get_direction(action)
        self.moving_point = Point(*np.clip(next_position, [0]*3, [self.size-1]*3))

        # Get observation, reward, terminated, truncated and info
        observation: np.ndarray = self.observer.get_observation(self.moving_point, self.target, self.map)
        reward: float = self.observer.get_reward(self.moving_point, self.target, self.map)
        terminated: bool = self.observer.is_terminated(self.moving_point, self.target, self.map)
        truncated: bool = self.observer.is_truncated(self.moving_point, self.target, self.map)
        info: dict = self.observer.get_info(self.moving_point, self.target, self.map)

        return (observation, reward, terminated, truncated, info)

    def render(self) -> None:
        if not self.window:
            self.graphics.init(self.size)
            self.window = True

        self.graphics.render(self.moving_point, self.target, self.map)
        self.graphics.set_title(f'{self.episode_steps}')
        self.graphics.update(RENDER_FPS)
