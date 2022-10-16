from tqdm import tqdm
import numpy as np

from src import *
from src.core.entities import ManipulatorData, Point, Axes, Vector
from src.communication import MockRepository, MujocoRepository


def main():
    Logger.info('program initialized')

    # Agent and env
    env = make_env('grid_world')
    agent = get_agent('deep_q_agent', size=(env.observation_space_size,
                                            64,
                                            env.action_space_size))

    # Training constants
    EPISODES: int = 10
    MAX_STEPS: int = 500
    SHOW_AFTER: int = 9

    # Initialization
    graphics: Graphics = Graphics()
    logging: Logging = Logging()
    repository: MockRepository = MockRepository()
    # repository: MujocoRepository = MujocoRepository('ur3')

    # Initialize manipulator and target
    initial_manipulator_data: ManipulatorData = Manipulator.create('ur3')

    initial_target: Point = Point(0.5, 0.15, 1.1)
    initial_target_axes: Axes = Axes(Vector(-1.0, 0.0, 0.0),
                                     Vector(0.0, 0.0, -1.0),
                                     Vector(0.0, -1.0, 0.0))

    # Initialize manipulator with repository
    repository.send_manipulator_data(initial_manipulator_data)
    manipulator_data = repository.get_manipulator_data()

    Logger.info('manipulator initialized')

    # Plot initial configuration
    graphics.show_initial_configuration(manipulator_data,
                                        initial_target,
                                        initial_target_axes)

    Logger.info('training model')
    # Train model
    for episode in tqdm(range(1, EPISODES+1)):
        # Train agent on env

        # Varibable resetting
        done: bool = False
        episode_step: int = 0
        observation, info = env.reset(return_info=True)

        while not done:
            # Check max steps in episode
            if episode_step >= MAX_STEPS:
                break

            # Render
            if episode >= SHOW_AFTER:
                graphics.show_training(info['virtual_point'], info['target'])

            # Get action from policy
            action: int = agent.get_action(observation)

            # Step on the environment dynamics
            new_observation, reward, done, info = env.step(action)

            # Train agent
            agent.train(observation, new_observation, action, reward, done)

            # Update variables
            observation = new_observation
            episode_step += 1

        env.close()
        agent.save_model()

    # Trained model
    for i in range(500):
        # Get new target
        target: Point = Movement.linear_y(i, height=0.8, distance=0.3)
        target_axes: Axes = manipulator_data.axes_list[-1]

        # Render
        graphics.show_current_state(manipulator_data,
                                    target,
                                    target_axes)

        # Perform Inverse Kinematics
        manipulator_data = Kinematics.fabrik_vanilla(manipulator_data,
                                                     target,
                                                     target_axes,
                                                     iterations=20,
                                                     update_axes=True)

        # Send manipulator data
        repository.send_manipulator_data(manipulator_data)
        manipulator_data = repository.get_manipulator_data()

        # Log data
        logging.angles_to_csv(manipulator_data.angles)
        logging.position_to_csv(manipulator_data.positions[-1])
        logging.target_to_csv(target)

    # Close graphics
    graphics.close()

    # Plot results
    Results.show_angles_plot()
    Results.show_path_plot()


if __name__ == '__main__':
    main()
