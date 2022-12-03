from ..core.entities import Point


class Target:
    @staticmethod
    def get():
        target: Point = Point(0.5, 0.2, 1.3)

        return target
