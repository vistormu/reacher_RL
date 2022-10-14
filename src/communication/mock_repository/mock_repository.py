import numpy as np

from .use_cases import Kinematics, ManipulatorDataGetter

from ...features.manipulator.repository.manipulator_repository import ManipulatorRepository
from src.core.entities import ManipulatorData


class MockRepository(ManipulatorRepository):
    def __init__(self) -> None:
        self._manipulator_data: ManipulatorData = None

    def send_manipulator_data(self, manipulator_data: ManipulatorData) -> None:
        # Perform forward kinematics with given angles
        transformation_matrices: list[np.ndarray] = Kinematics.forward(
            manipulator_data)

        # Update manipulator data
        self._manipulator_data = ManipulatorDataGetter.get(
            transformation_matrices, manipulator_data)

    def get_manipulator_data(self) -> ManipulatorData:
        return self._manipulator_data
