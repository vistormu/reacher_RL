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
    manipulator.init(base, angles=[np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0])
    manipulator.data.ik_parameters.base_to = 3

    # 4. Get target
    target: Target = Target()
    target_position: Point = target.get()

    # 5. Calculate path
    Logger.info('getting path')
    end_effector: Point = manipulator.data.systems[-1].position
    path: Optional[list[Point]] = path_planner.get_path(end_effector, target_position, render=False)
    while path is None:
        Logger.warning('path not found')
        path: Optional[list[Point]] = path_planner.get_path(end_effector, target_position)

    # path: list[Point] = path_planner.mock_path(step=2, height=1.1, distance=0.5)

    # 5. Get the angles path
    Logger.info('getting angles path')
    angles_path: list[list[float]] = manipulator.follow_path(path, target_position, render=True)

    # 6. Send data to the real manipulator
    answer: str = input('send data? [y/n]: ')
    if answer == 'y':
        Logger.warning('you f**** up')
        logging.log_path(path, angles_path)
        logging.plot_results()


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--train',
                        action='store_true',
                        help='call to train the model')

    main(vars(parser.parse_args()))
