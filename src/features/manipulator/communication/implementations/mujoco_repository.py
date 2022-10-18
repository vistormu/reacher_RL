import mujoco_py
import numpy as np

from copy import deepcopy

from ...entities import ManipulatorData
from ....core.entities import Point, Axes, Vector
from ...repository import ManipulatorRepository

ASSETS_PATH = 'src/assets/'


class MujocoRepository(ManipulatorRepository):
    def __init__(self, robot: str) -> None:
        self._manipulator_data: ManipulatorData = None

        path: str = f'{ASSETS_PATH}{robot}/robot.xml'
        self.model = mujoco_py.load_model_from_path(path)
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjViewer(self.sim)

    def send_manipulator_data(self, manipulator_data: ManipulatorData) -> None:
        self._manipulator_data = manipulator_data

        self.sim.data.qpos[:] = manipulator_data.angles
        self.sim.step()
        self.viewer.render()

    def get_manipulator_data(self) -> ManipulatorData:
        new_manipulator_data: ManipulatorData = deepcopy(
            self._manipulator_data)

        new_manipulator_data.angles = list(self.sim.data.qpos)
        new_manipulator_data.positions = self._get_positions(
            self.sim.data.body_xpos)
        new_manipulator_data.axes_list = self._get_axes_list(
            self.sim.data.body_xmat)

        return new_manipulator_data

    @staticmethod
    def _get_positions(positions: np.ndarray) -> list[Point]:
        new_positions: list[Point] = []
        forbidden_points: list[int] = [0, 2, 9, 10]
        for i, position in enumerate(positions):
            if i in forbidden_points:
                continue

            new_position: Point = Point(-position[0], -
                                        position[1], position[2])

            new_positions.append(new_position)

        return new_positions

    @staticmethod
    def _get_axes_list(orientations: np.ndarray) -> list[Axes]:
        new_axes_list: list[Axes] = []
        for i, orientation in enumerate(orientations):
            if i == 1:
                x_axis: Vector = Vector(
                    orientation[0], orientation[1], orientation[2])
                y_axis: Vector = Vector(
                    orientation[3], orientation[4], orientation[5])
                z_axis: Vector = Vector(
                    orientation[6], orientation[7], orientation[8])
            elif i == 2:
                x_axis: Vector = Vector(
                    orientation[0], orientation[3], orientation[6])
                y_axis: Vector = Vector(
                    -orientation[2], -orientation[5], orientation[8])
                z_axis: Vector = Vector(
                    -orientation[1], -orientation[4], orientation[7])
            elif i == 3 or i == 4:
                x_axis: Vector = Vector(
                    orientation[2], orientation[5], -orientation[8])
                y_axis: Vector = Vector(
                    orientation[0], orientation[3], -orientation[6])
                z_axis: Vector = Vector(
                    -orientation[1], -orientation[4], orientation[7])
            elif i == 5:
                x_axis: Vector = Vector(
                    -orientation[0], -orientation[3], orientation[6])
                y_axis: Vector = Vector(
                    -orientation[1], -orientation[4], orientation[7])
                z_axis: Vector = Vector(
                    -orientation[2], -orientation[5], orientation[8])
            elif i == 6 or i == 7:
                x_axis: Vector = Vector(
                    -orientation[0], -orientation[3], orientation[6])
                y_axis: Vector = Vector(
                    orientation[2], orientation[5], -orientation[8])
                z_axis: Vector = Vector(
                    -orientation[1], -orientation[4], orientation[7])
            else:
                continue

            new_axes: Axes = Axes(x_axis, y_axis, z_axis)

            new_axes_list.append(new_axes)

        return new_axes_list
