import numpy as np
from src.core.entities import Point


class Movement:
    @staticmethod
    def linear_y(time_stamp: int, step: float = 1.0, height: float = 1.10, distance: float = 0.5) -> Point:
        x: float = distance
        y: float = np.sin(time_stamp*step/180.0*np.pi)*0.15 + 0.15
        z: float = height

        return Point(x, y, z)
