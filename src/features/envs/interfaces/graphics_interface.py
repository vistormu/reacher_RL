from abc import ABC, abstractmethod


class IGraphics(ABC):
    @abstractmethod
    def init(self) -> None:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    def update(self, fps: int) -> None:
        pass

    def close(self) -> None:
        pass
