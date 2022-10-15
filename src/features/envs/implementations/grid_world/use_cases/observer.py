import numpy as np

from src.core.entities import Point
from ....interfaces import IObserver


class Observer(IObserver):
    def __init__(self, size: float, limits: list[tuple[float, float]]) -> None:
        self.previous_distance: float = 0.0

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

    def get_observation(self, virtual_point: Point, target: Point) -> np.ndarray:
        return np.array([
            virtual_point.x,
            virtual_point.y,
            virtual_point.z,
            target.x,
            target.y,
            target.z,
            virtual_point.x - target.x,
            virtual_point.y - target.y,
            virtual_point.y - target.z,
        ])

    def get_reward(self, virtual_point: Point, target: Point) -> int:
        pass

    def get_info(self, virtual_point: Point, target: Point) -> dict:
        return dict()

    def is_done(self, virtual_point: Point, target: Point) -> bool:
        pass
