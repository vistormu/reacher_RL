import argparse

from src import *
from src.features.core.entities import Point


def main(args):
    Logger.info('program initialized')

    # 1. Train model
    training: bool = False
    if args['train']:
        training = True

    if training:
        training: Training = Training()
        training.train()

    # 2. Get path from agent
    # WIP
    path: list[Point] = PathGetter.mock_movement()

    # 3. Choose a good manipulator orientation
    # WIP
    # Entities
    manipulator: Manipulator = Manipulator('ur3', 'mock')
    logging: Logging = Logging()

    # 4. Get the angles path
    for point in path:
        # Move manipulator
        manipulator.move_to(point)

        # Render
        manipulator.render(30)

        # Log results
        logging.log_manipulator_data(manipulator.manipulator_data)
        logging.log_path(point)

    # Closing
    manipulator.close()

    # Plot results
    logging.show_results()

    # 5. Send data to the real manipulator
    # WIP
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
