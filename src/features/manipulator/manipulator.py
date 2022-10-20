from ..logging import Logging

from .entities import ManipulatorData
from ..core.entities import Point

from .use_cases import ManipulatorDataParser, Graphics, InverseKinematics
from .communication import get_repository
from .repository import ManipulatorRepository

ASSETS_PATH: str = "src/assets/"


class Manipulator:
    def __init__(self, robot: str, repository_id: str) -> None:
        # Manipulator data
        self.robot: str = robot
        self.manipulator_data: ManipulatorData = ManipulatorDataParser.get(robot)

        # Use cases
        self.graphics: Graphics = Graphics()

        # Repository
        self.repository: ManipulatorRepository = get_repository(repository_id)
        self.repository.send_manipulator_data(self.manipulator_data)
        self.manipulator_data = self.repository.get_manipulator_data()

        # Logging
        self.logging: Logging = Logging()

    def follow_path(self, path: list[Point], target: Point) -> None:
        for virtual_point in path:
            self._move_to(virtual_point)

            self.graphics.render(self.manipulator_data,
                                 virtual_point,
                                 target,
                                 path)
            self.graphics.update(10)

            self.logging.log_manipulator_data(self.manipulator_data.angles,
                                              self.manipulator_data.positions)
            self.logging.log_path(virtual_point)

        self.graphics.close()

    def _move_to(self, target: Point) -> None:
        # Apply inverse kinematics
        self.manipulator_data = InverseKinematics.fabrik_vanilla(self.manipulator_data,
                                                                 target,
                                                                 self.manipulator_data.axes_list[-1],
                                                                 update_axes=True)

        # Send manipulator data (Subject to change)
        self.repository.send_manipulator_data(self.manipulator_data)
        self.manipulator_data = self.repository.get_manipulator_data()

    def send(self) -> None:
        return
