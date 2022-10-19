import numpy as np
import bgplot as bgp

from ......core.entities import Point
from ....interfaces import IObserver


class Observer(IObserver):
    MOVE_PENALTY = 1
    TARGET_REACHED_REWARD = 10

    def __init__(self, size: float, limits: list[tuple[float, float]]) -> None:
        # Observation space
        self.size: float = size
        self.limits: list[tuple[float, float]] = limits

        self.low: np.ndarray = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            -1.0,
            -1.0,
        ]).astype(np.float32)

        self.high: np.ndarray = np.array([
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]).astype(np.float32)

        self.observation_space_size: int = len(self.low)

        # Reward variables
        self.previous_distance: float = 0.0

    def get_observation(self, virtual_point: Point, target: Point) -> np.ndarray:
        normalized_virtual_point: Point = self._normalize_point(virtual_point)
        normalized_target: Point = self._normalize_point(target)

        return np.array([
            normalized_virtual_point.x,
            normalized_virtual_point.y,
            normalized_virtual_point.z,
            normalized_target.x,
            normalized_target.y,
            normalized_target.z,
            normalized_virtual_point.x - normalized_target.x,
            normalized_virtual_point.y - normalized_target.y,
            normalized_virtual_point.y - normalized_target.z,
        ])

    def get_reward(self, virtual_point: Point, target: Point) -> int:
        reward: int = 0

        # Penalty for moving
        reward -= self.MOVE_PENALTY

        # Penalty for not reducing distance
        distance: float = bgp.ops.distance_between_two_points(
            virtual_point, target)
        if distance >= self.previous_distance:
            reward -= self.MOVE_PENALTY*2

        self.previous_distance = distance

        # Reward for reaching the objective
        if self._is_equal(virtual_point, target):
            reward += self.TARGET_REACHED_REWARD

        return reward

    def get_info(self, virtual_point: Point, target: Point) -> dict:
        return {'virtual_point': virtual_point,
                'target': target}

    def is_done(self, virtual_point: Point, target: Point) -> bool:
        return self._is_equal(virtual_point, target)

    def _is_equal(self, virtual_point: Point, target: Point, factor: float = 1.0) -> bool:
        return np.all(np.abs(np.array(virtual_point - target)) <= (self.size*factor))

    def _normalize_point(self, point: Point) -> Point:
        x: float = self._normalize_interval(
            self.limits[0][0], self.limits[0][1], point.x)
        y: float = self._normalize_interval(
            self.limits[1][0], self.limits[1][1], point.y)
        z: float = self._normalize_interval(
            self.limits[2][0], self.limits[2][1], point.z)

        return Point(x, y, z)

    @staticmethod
    def _normalize_interval(low: float, high: float, value: float) -> float:
        return 1/(high-low)*(value-low)
