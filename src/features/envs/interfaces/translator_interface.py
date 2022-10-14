from abc import ABC


class ITranslator(ABC):
    def __init__(self) -> None:
        self.action_space_size: int
