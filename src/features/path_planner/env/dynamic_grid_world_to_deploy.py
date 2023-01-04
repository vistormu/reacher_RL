import numpy as np
from typing import Optional

from .use_cases import Observer, Translator, Graphics
from .entities import Point, OccupancyGrid
from ...core.entities import Point as BgpPoint

RENDER_FPS: int = 10


class DynamicGridworldToDeploy:
    def __init__(self, size: int) -> None:
        # Variables
        self.size: int = size

        # Observer and translator
        self.observer: Observer = Observer(size)
        self.translator: Translator = Translator()
        self.graphics: Graphics = Graphics()

        # Entities
        self.moving_point: Point = Point(0, 0, 0)
        self.target: Point = Point(0, 0, 0)
        self.map: Optional[OccupancyGrid] = None

        # Other
        self.window: bool = False

    def init(self, bgp_moving_point: BgpPoint, bgp_target: BgpPoint, map: Optional[np.ndarray]) -> tuple[np.ndarray, dict]:
        # From dimensional to adimensional
        distance_vector: BgpPoint = bgp_target - bgp_moving_point
        grid_size: float = np.max(np.abs(distance_vector))/self.size
        self.grid_size: float = grid_size

        x_moving_point: int = 0 if distance_vector.x > 0 else int(abs(distance_vector.x)/grid_size)
        y_moving_point: int = 0 if distance_vector.y > 0 else int(abs(distance_vector.y)/grid_size)
        z_moving_point: int = 0 if distance_vector.z > 0 else int(abs(distance_vector.z)/grid_size)
        self.moving_point = Point(x_moving_point, y_moving_point, z_moving_point)

        x_target: int = 0 if distance_vector.x < 0 else int(abs(distance_vector.x)/grid_size)
        y_target: int = 0 if distance_vector.y < 0 else int(abs(distance_vector.y)/grid_size)
        z_target: int = 0 if distance_vector.z < 0 else int(abs(distance_vector.z)/grid_size)
        self.target = Point(x_target, y_target, z_target)

        print('TARGET:', bgp_target, self.target)
        print('MOVING_POINT:', bgp_moving_point, self.moving_point)

        # Init occupancy grid
        if map is not None:
            self.map = OccupancyGrid(self.size, seed=0)
            assert map.shape[0] == self.size
            self.map.map = map

        # Get observation and info
        self.observer.update(self.moving_point, self.target, self.map)
        observation: np.ndarray = self.observer.get_observation()
        info: dict = self.observer.get_info()

        return (observation, info)

    def step(self, action) -> tuple[np.ndarray, bool, bool, dict]:
        # Update moving point
        next_position: Point = self.translator.get_direction(action)
        self.moving_point += next_position

        # Next position to bgp
        bgp_next_position: BgpPoint = BgpPoint(*next_position)*self.grid_size

        # Get observation, reward, terminated, truncated and info
        self.observer.update(self.moving_point, self.target, self.map)
        observation: np.ndarray = self.observer.get_observation()
        terminated: bool = self.observer.is_terminated()
        truncated: bool = self.observer.is_truncated()
        info: dict = self.observer.get_info()
        info['next_position'] = bgp_next_position

        return (observation, terminated, truncated, info)

    def render(self) -> None:
        if not self.window:
            self.graphics.init(self.size)
            self.window = True

        self.graphics.render(self.moving_point, self.target, self.map)
        # self.graphics.set_title()
        self.graphics.update(RENDER_FPS)
