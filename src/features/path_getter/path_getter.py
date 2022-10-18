import numpy as np

from ..core.entities import Point


class PathGetter:
    @staticmethod
    def mock_movement(step: float = 1.0, height: float = 1.10, distance: float = 0.5) -> list[Point]:
        path: list[Point] = []

        for i in range(220):
            x: float = distance
            y: float = np.sin(i*step/180.0*np.pi)*0.15 + 0.15
            z: float = height

            path.append(Point(x, y, z))

        return path
