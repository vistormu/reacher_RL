import argparse

from src import *
from src.features.core.entities import Point


def main(args):
    Logger.info('program initialized')

    path_planner: PathPlanner = PathPlanner('grid_world', 'deep_q_agent')
    # 1. Train model
    if args['train']:
        path_planner.train()

    manipulator: Manipulator = Manipulator('ur3', 'mock')
    logging: Logging = Logging()

    # 2. Choose a good manipulator orientation
    # WIP

    # 3. Get path from agent
    # path: list[Point] = PathPlanner.mock_path()
    Logger.info('getting path')
    path: list[Point] = path_planner.get_path(virtual_point=manipulator.manipulator_data.positions[-1],
                                              target=Point(0.5, 0.3, 1.1))

    # 4. Get the angles path
    Logger.info('getting angles path')
    manipulator.follow_path(path)

    # Plot results
    logging.show_results()

    # 5. Send data to the real manipulator
    answer: str = input('send data? [y/n]: ')
    if answer == 'y':
        manipulator.send()
        Logger.warning('you f**** up')


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--train',
                        action='store_true',
                        help='call to train the model')

    main(vars(parser.parse_args()))
