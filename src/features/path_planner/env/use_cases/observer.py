import numpy as np

from typing import Optional

from ..interfaces import IObserver
from ..entities import Point, OccupancyGrid


class Observer(IObserver):
    MOVE_PENALTY: int = -1
    DISTANCING_PENALTY: int = -10
    COLLISION_PENALTY: int = -20
    TARGET_REACHED_REWARD: int = 100

    def __init__(self, size: int) -> None:
        self.size: int = size
        self.observation_space_size = 4

        # Reward variables
        self.previous_distance: int = 0
        self.previous_distance_vector: Point = Point(0, 0, 0)

        # Observer variables
        self.moving_point: Point = Point(0, 0, 0)
        self.target: Point = Point(0, 0, 0)
        self.map: Optional[OccupancyGrid] = None
        self.next_direction: Point = Point(0, 0, 0)

    def update(self, moving_point: Point, target: Point, map: Optional[OccupancyGrid]) -> None:
        self.moving_point = moving_point
        self.target = target
        self.map = map

    def get_observation(self) -> np.ndarray:
        distance_to_target: Point = self.target - self.moving_point
        if self.map is not None:
            danger_index: float = self._get_danger_index(self.moving_point, self.map)
        else:
            danger_index: float = self.moving_point.z/self.size

        return np.array([
            distance_to_target.x/self.size,
            distance_to_target.y/self.size,
            distance_to_target.z/self.size,
            danger_index,
        ])

    def get_reward(self) -> float:
        reward: float = 0.0

        # Penalty for moving
        reward += self.MOVE_PENALTY

        # Penalty for increasing the distance to the target
        distance_vector: Point = Point(*np.abs(self.target-self.moving_point))
        reduced_in_x: bool = True if distance_vector.x < self.previous_distance_vector.x else False
        reduced_in_y: bool = True if distance_vector.y < self.previous_distance_vector.y else False
        reduced_in_z: bool = True if distance_vector.z < self.previous_distance_vector.z else False

        if (not reduced_in_x) and (not reduced_in_y) and (not reduced_in_z):
            reward += self.DISTANCING_PENALTY

        self.previous_distance_vector = distance_vector

        # Penalty for colliding
        if self.map is not None:
            if self._is_collision(self.moving_point, self.map):
                reward += self.COLLISION_PENALTY

        if self._is_wall_collision(self.moving_point):
            reward += self.COLLISION_PENALTY

        # Reward for reaching the target
        if self.moving_point == self.target:
            reward += self.TARGET_REACHED_REWARD

        return reward

    def get_info(self) -> dict:
        return {}

    def is_terminated(self) -> bool:
        return self.moving_point == self.target

    def is_truncated(self) -> bool:
        if self.map is not None:
            return self._is_collision(self.moving_point, self.map)
        else:
            return False

    def _get_danger_index(self, moving_point: Point, map: OccupancyGrid) -> float:
        x: int = moving_point.x
        y: int = moving_point.y

        k: int = int(map.size*0.1)

        i_lower: int = np.clip(x-k, 0, map.size)
        i_upper: int = np.clip(x+k, 0, map.size)
        j_lower: int = np.clip(y-k, 0, map.size)
        j_upper: int = np.clip(y+k, 0, map.size)

        mean_surrounding_heights: float = np.mean(map.map[i_lower:(i_upper+1), j_lower:(j_upper)+1]).astype(float)
        danger_index: float = (moving_point.z - mean_surrounding_heights)/self.size

        return danger_index

    @staticmethod
    def _is_collision(moving_point: Point, map: OccupancyGrid) -> bool:
        return True if moving_point.z <= map.get_value(moving_point.x, moving_point.y) else False

    def _is_wall_collision(self, moving_point: Point) -> bool:
        return bool(np.any(moving_point == Point(0, 0, 0))) or bool(np.any(moving_point == Point(self.size, self.size, self.size)))
