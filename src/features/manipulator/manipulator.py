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

        # Target
        self.target: Point = None

        # Use cases
        self.graphics: Graphics = Graphics()

        # Repository
        self.repository: ManipulatorRepository = get_repository(repository_id)
        self.repository.send_manipulator_data(self.manipulator_data)
        self.manipulator_data = self.repository.get_manipulator_data()

    def move_to(self, target: Point) -> None:
        # Update target
        self.target = target

        # Apply inverse kinematics
        self.manipulator_data = InverseKinematics.fabrik_vanilla(self.manipulator_data,
                                                                 target,
                                                                 self.manipulator_data.axes_list[-1],
                                                                 update_axes=True)

        # Send manipulator data (Subject to change)
        self.repository.send_manipulator_data(self.manipulator_data)
        self.manipulator_data = self.repository.get_manipulator_data()

    def render(self, fps: int) -> None:
        self.graphics.render(self.manipulator_data,
                             self.target)
        self.graphics.update(fps)

    def close(self):
        self.graphics.close()
