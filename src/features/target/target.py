from ..core.entities import Point

from .repository import TargetRepository
from .communication import MockTargetRepository


class Target:
    def __init__(self) -> None:
        self.repository: TargetRepository = MockTargetRepository()

    def get(self) -> Point:
        return self.repository.get()
