from copy import deepcopy

import bgplot as bgp
from bgplot.entities import Point, Axes

from .algorithms import FabrikStep, FabrikVanilla
from src.core.entities import ManipulatorData


class Kinematics:
    @staticmethod
    def fabrik_step(manipulator_data: ManipulatorData, target: Point, target_axes: Axes, iterations: int = 10, threshold: float = 0.001) -> ManipulatorData:
        # best_manipulator_data: ManipulatorData = deepcopy(manipulator_data)
        # best_distance: float = bgp.ops.distance_between_two_points(manipulator_data.positions[-1],
        #                                                            target)
        for _ in range(iterations):
            manipulator_data = FabrikStep.iterate(manipulator_data,
                                                  target,
                                                  target_axes)

            distance: float = bgp.ops.distance_between_two_points(manipulator_data.positions[-1],
                                                                  target)

            # if distance < best_distance:
            #     best_distance = distance
            #     best_manipulator_data = deepcopy(manipulator_data)

            # debug
            target_axes = manipulator_data.axes_list[-1]

            if distance < threshold:
                break

        return manipulator_data

    @staticmethod
    def fabrik_vanilla(manipulator_data: ManipulatorData, target: Point, target_axes: Axes, iterations: int = 10, threshold: float = 0.001, update_axes: bool = False) -> ManipulatorData:
        # best_manipulator_data: ManipulatorData = deepcopy(manipulator_data)
        # best_distance: float = bgp.ops.distance_between_two_points(manipulator_data.positions[-1],
        #                                                            target)
        for _ in range(iterations):
            manipulator_data = FabrikVanilla.iterate(manipulator_data,
                                                     target,
                                                     target_axes)

            distance: float = bgp.ops.distance_between_two_points(manipulator_data.positions[-1],
                                                                  target)

            # if distance < best_distance:
            #     best_distance = distance
            #     best_manipulator_data = deepcopy(manipulator_data)

            if update_axes:
                target_axes = manipulator_data.axes_list[-1]

            if distance < threshold:
                break

        # return best_manipulator_data
        return manipulator_data
