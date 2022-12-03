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
        path_planner.train()

    # Initialize manipulator
    base: OrientedPoint = OrientedPoint(Point(0.0, 0.0, 0.0), Axes(Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0)))
    manipulator: Manipulator = Manipulator('ur3', 'mock')
    manipulator.init(base, angles=[np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Initialize logging
    logging: Logging = Logging()

    # 4. Get target
    target: Point = Target.get()

    # 5. Calculate path
    Logger.info('getting path')
    path: list[Point] = path_planner.get_path(virtual_point=manipulator.data.systems[-1].position,
                                              target=target)

    # 5. Get the angles path
    Logger.info('getting angles path')
    manipulator.follow_path(path, target)

    # Plot results
    logging.show_results()

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
