from abc import ABC, abstractmethod

from ...core.entities import Point


class TargetRepository:
    @abstractmethod
    def get(self) -> Point:
        pass
