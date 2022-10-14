from abc import ABC, abstractmethod

from src.core.entities import ManipulatorData


class ManipulatorRepository(ABC):

    @abstractmethod
    def get_manipulator_data(self) -> ManipulatorData:
        pass

    @abstractmethod
    def send_manipulator_data(self, manipulator_data: ManipulatorData) -> None:
        pass
