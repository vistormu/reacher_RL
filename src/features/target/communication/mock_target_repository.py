from ..repository import TargetRepository

from ...core.entities import Point


class MockTargetRepository(TargetRepository):
    def get(self) -> Point:
        return Point(0.4, 0.3, 1.1)
