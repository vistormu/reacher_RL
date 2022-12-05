import bgplot as bgp

from .entities import ManipulatorData, DHParameters, IKParameters
from ..core.entities import Point, OrientedPoint

from .use_cases import ParametersManager, Graphics, Farm
from .communication import get_repository
from .repository import ManipulatorRepository

TOLERANCE: float = 0.005
MAX_ITERATIONS: int = 20


class Manipulator:
    def __init__(self, robot: str, repository_id: str) -> None:
        # Manipulator data
        self.robot: str = robot
        self.data: ManipulatorData = None  # type: ignore

        # Use cases
        self.graphics: Graphics = Graphics()

        # Repository
        self.repository: ManipulatorRepository = get_repository(repository_id)

    def init(self, base: OrientedPoint, angles: list[float]):
        # Initialize DH Parameters
        dh_parameters: DHParameters = ParametersManager.get_dh_parameters(self.robot)

        # The angles must be the same length as the DoFs
        assert len(angles) == len(dh_parameters.a)

        # Initialize angles
        angles.insert(0, 0.0)

        # Extend the DH Parameters to include the base
        extended_dh_parameters: DHParameters = ParametersManager.extend_dh_parameters(dh_parameters, base)

        # Initialize the IK Parameters
        ik_parameters: IKParameters = ParametersManager.get_ik_parameters(extended_dh_parameters)

        # Initialize the systems
        systems: list[OrientedPoint] = []

        self.data = ManipulatorData(extended_dh_parameters, ik_parameters, systems, angles)

        # TMP
        self._send()

    def follow_path(self, path: list[Point], target: Point) -> None:
        for virtual_point in path:
            self._move_to(virtual_point)
            self._send()

            self.graphics.render(self.data,
                                 virtual_point,
                                 target,
                                 path)
            self.graphics.update(10)

        self.graphics.close()

    def _move_to(self, target: Point) -> None:
        distance: float = bgp.ops.distance_between_two_points(target, self.data.systems[-1].position)
        iterations: int = 0
        while distance > TOLERANCE:
            if iterations == MAX_ITERATIONS:
                break

            self.data = Farm.iterate(self.data, OrientedPoint(target, self.data.systems[-1].axes))

            distance = bgp.ops.distance_between_two_points(target, self.data.systems[-1].position)
            iterations += 1

    def _send(self) -> None:
        self.repository.send_manipulator_data(self.data)
        self.data = self.repository.get_manipulator_data()
