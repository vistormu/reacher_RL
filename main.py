import argparse
import numpy as np

from typing import Optional

from src import *
from src.features.core.entities import Point, OrientedPoint, Axes, Vector


def main(args):
    Logger.info('program initialized')

    # Variables
    logging: Logging = Logging()

    # Initialize path planner
    path_planner: PathPlanner = PathPlanner()
    # 1. Train model
    if args['train']:
        path_planner.train()

    # Initialize manipulator
    base: OrientedPoint = OrientedPoint(Point(0.07, 0.13, 1.15), Axes(Vector(1.0, 0.0, 0.0), Vector(0.0, 0.7071, -0.7071), Vector(0.0, 0.7071, 0.7071)))
    # base: OrientedPoint = OrientedPoint(Point(0.0, 0.0, 0.0), Axes(Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0)))
    manipulator: Manipulator = Manipulator('ur3', 'mock')
    manipulator.init(base, angles=[-np.pi/2, np.pi/4, 0.0, -np.pi/2, np.pi, 0.0])
    manipulator.data.ik_parameters.base_to = 3

    # 4. Get target
    # target: Target = Target()
    # target_position: Point = target.get()

    target_list: list[Point] = [Point(0.25, 0.1, 0.85), Point(0.25, 0.2, 0.85),  Point(0.2, 0.3, 0.85)]
    occupancy_map: np.ndarray = np.zeros((50, 50))
    occupancy_map[30:, :] = 0.5*50

    full_path: list[Point] = []
    full_angles_path: list[list[float]] = []

    for target in target_list:
        # 5. Calculate path
        Logger.info('getting path')
        end_effector: Point = manipulator.data.systems[-1].position
        path: Optional[list[Point]] = path_planner.get_path(end_effector, target, occupancy_map, render=True)

        if path is None:
            Logger.warning('path not found')
            return
        else:
            full_path += path
            # path: list[Point] = path_planner.mock_path(step=2, height=1.1, distance=0.5)

        # 6. Get the angles path
        Logger.info('getting angles path')
        angles_path: list[list[float]] = manipulator.follow_path(path, target, render=True)
        full_angles_path += angles_path

        # Go back to the initial point
        return_path: list[Point] = list(reversed(path))
        return_angles_path: list[list[float]] = manipulator.follow_path(return_path, return_path[-1], render=True)

        full_path += return_path
        full_angles_path += return_angles_path

    # 6. Send data to the real manipulator
    answer: str = input('send data? [y/n]: ')
    if answer == 'y':
        Logger.warning('you f**** up')
        logging.log_path(full_path, full_angles_path)
        logging.plot_results()


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--train',
                        action='store_true',
                        help='call to train the model')

    main(vars(parser.parse_args()))
