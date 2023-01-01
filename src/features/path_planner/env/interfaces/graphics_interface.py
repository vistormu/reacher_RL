from abc import ABC, abstractmethod


class IGraphics(ABC):
    @abstractmethod
    def init(self) -> None:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def update(self, fps: int) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
