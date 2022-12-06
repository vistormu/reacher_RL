import argparse
import numpy as np

from src import *
from src.features.core.entities import Point, OrientedPoint, Axes, Vector


def main(args):
    Logger.info('program initialized')

    # Initialize path planner
    path_planner: PathPlanner = PathPlanner('grid_world', 'deep_q_agent')
    # 1. Train model
    if args['train']:
        size_list: list[float] = [0.2, 0.15, 0.1, 0.075, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01]
        for size in size_list:
            path_planner.train(episodes=200, size=size)

    # Initialize manipulator
    # base: OrientedPoint = OrientedPoint(Point(0.07, 0.13, 1.15), Axes(Vector(1.0, 0.0, 0.0), Vector(0.0, 0.7071, -0.7071), Vector(0.0, 0.7071, 0.7071)))
    base: OrientedPoint = OrientedPoint(Point(0.0, 0.0, 0.0), Axes(Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0)))
    manipulator: Manipulator = Manipulator('ur3', 'mock')
    manipulator.init(base, angles=[np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
    manipulator.data.ik_parameters.base_to = 3  # TMP

    # 4. Get target
    target: Target = Target()
    target_position: Point = target.get()

    # 5. Calculate path
    Logger.info('getting path')
    path: list[Point] = path_planner.get_path(manipulator.data.systems[-1].position, target_position)
    # path: list[Point] = path_planner.mock_path(step=2, height=0.1, distance=0.5)

    # 5. Get the angles path
    Logger.info('getting angles path')
    manipulator.follow_path(path, target_position)

    # 6. Send data to the real manipulator
    answer: str = input('send data? [y/n]: ')
    if answer == 'y':
        Logger.warning('you f**** up')


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--train',
                        action='store_true',
                        help='call to train the model')

    main(vars(parser.parse_args()))
