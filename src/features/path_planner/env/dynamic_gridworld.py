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

        random_number: int = np.random.randint(0, 2)

        # Place target
        if self.map is not None:
            x, y = self.map.get_max_index()
            z = self.map.get_value(x, y) + 1
        else:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            z = np.random.randint(0, self.size)

        if random_number:
            self.moving_point = Point(x, y, z)
        else:
            self.target = Point(x, y, z)

        # Place moving point
        if self.map is not None:
            x, y = self.map.get_min_index()
            z = self.map.get_value(x, y) + 1
        else:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            z = np.random.randint(0, self.size)

        if random_number:
            self.target = Point(x, y, z)
        else:
            self.moving_point = Point(x, y, z)

        # Get observation and info
        self.observer.update(self.moving_point, self.target, self.map)
        observation: np.ndarray = self.observer.get_observation()
        info: dict = self.observer.get_info()

        return (observation, info)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.episode_steps += 1

        # Update moving point
        next_position: Point = self.moving_point + self.translator.get_direction(action)
        self.moving_point = Point(*np.clip(next_position, [0]*3, [self.size-1]*3))

        # Get observation, reward, terminated, truncated and info
        self.observer.update(self.moving_point, self.target, self.map)
        observation: np.ndarray = self.observer.get_observation()
        reward: float = self.observer.get_reward()
        terminated: bool = self.observer.is_terminated()
        truncated: bool = self.observer.is_truncated()
        info: dict = self.observer.get_info()

        return (observation, reward, terminated, truncated, info)

    def render(self) -> None:
        if not self.window:
            self.graphics.init(self.size)
            self.window = True

        self.graphics.render(self.moving_point, self.target, self.map)
        self.graphics.set_title(f'{self.episode_steps}')
        self.graphics.update(RENDER_FPS)
