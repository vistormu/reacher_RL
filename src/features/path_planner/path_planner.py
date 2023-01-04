import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.signal import savgol_filter
from typing import Optional

from ...core import Logger
from ..core.entities import Point
from .agent import DQN, TrainedDQN
from .env import DynamicGridworld, DynamicGridworldToDeploy

LOGGING_PATH: str = 'src/features/logging/data/training/'


class PathPlanner:
    def __init__(self) -> None:
        Path(LOGGING_PATH).mkdir(parents=True, exist_ok=True)

    def train(self, *, size: int = 50, episodes: int = 10000, obstacles_enabled: bool = True, render: bool = False) -> None:
        filename: str = f'obs_{size}' if obstacles_enabled else f'no_obs_{size}'

        # Logging info
        Logger.info('% ===============')
        Logger.info('  TRAINING MODEL')
        Logger.info('% ===============')
        Logger.info(f'obstacles enabled: {obstacles_enabled}')
        Logger.info(f'with grid size: {size}')
        Logger.info(f'using the model: {filename}.h5')
        Logger.info(f'filename generated: {filename}.csv')

        # Agent and env
        env: DynamicGridworld = DynamicGridworld(size, obstacles_enabled)
        agent: DQN = DQN(env.observation_space_size, env.action_space_size)

        # Training constants
        EPISODES: int = episodes
        MAX_STEPS: int = size*3
        SHOW_LAST: int = int(episodes*0.1) if render else -1

        try:
            agent.load_model(filename)
            Logger.info('model loaded')
        except:
            Logger.warning('model not found')

        # Variables
        times_terminated: int = 0
        times_truncated: int = 0
        max_steps_reached_count: int = 0

        terminated_list: list[bool] = []
        truncated_list: list[bool] = []
        not_finalized_list: list[bool] = []

        accumulated_reward_list: list[float] = []
        epsilon_list: list[float] = []
        episode_steps_list: list[int] = []
        loss_list: list[float] = []
        accuracy_list: list[float] = []

        # Train model
        for episode in tqdm(range(1, EPISODES+1)):
            # Variable resetting
            terminated: bool = False
            truncated: bool = False
            not_finalized: bool = False

            episode_step: int = 0
            accumulated_reward: float = 0.0

            # First observation
            observation, info = env.reset(seed=episode)

            # Wait until user input for rendering
            if episode == EPISODES-SHOW_LAST:
                input('press any key ot continue...')

            for step in range(MAX_STEPS):
                # Render
                if episode >= EPISODES-SHOW_LAST:
                    env.render()

                # Get action from policy
                action: int = agent.get_action(observation)

                # Step on the environment dynamics
                new_observation, reward, terminated, truncated, info = env.step(action)

                # Train agent
                agent.train(observation, new_observation, action, reward, terminated)

                # Update variables
                observation = new_observation
                episode_step += 1
                accumulated_reward += reward

                if step == MAX_STEPS-1:
                    max_steps_reached_count += 1
                    not_finalized = True

                # Break condition
                if truncated or terminated:
                    if terminated:
                        times_terminated += 1

                    if truncated:
                        times_truncated += 1

                    break

            # Save data to the lists
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            not_finalized_list.append(not_finalized)
            accumulated_reward_list.append(accumulated_reward)
            epsilon_list.append(agent.epsilon)
            episode_steps_list.append(episode_step)
            loss_list.append(agent.metrics[0])
            accuracy_list.append(agent.metrics[1])

        Logger.info(f'terminated percentage: {times_terminated}/{EPISODES} ({times_terminated*100//EPISODES}%)')
        Logger.info(f'truncated percentage: {times_truncated}/{EPISODES} ({times_truncated*100//EPISODES}%)')
        Logger.info(f'not finalized: {max_steps_reached_count}/{EPISODES} ({max_steps_reached_count*100//EPISODES}%)')

        # Save training data
        data: dict = {
            'accumulated_reward': accumulated_reward_list,
            'epsilon': epsilon_list,
            'terminated': terminated_list,
            'truncated': truncated_list,
            'not_finalized': not_finalized_list,
            'episode_steps': episode_steps_list,
            'loss': loss_list,
            'accuracy': accuracy_list,
        }

        self.save_data(filename, data)
        Logger.info('data saved')

        # Save model
        agent.save_model(filename)
        Logger.info('model saved')

    def get_path(self, moving_point: Point, target: Point, map: Optional[np.ndarray] = None, size: int = 50, max_steps: int = 200, render: bool = False) -> Optional[list[Point]]:
        # Env and agent
        env: DynamicGridworldToDeploy = DynamicGridworldToDeploy(size)
        agent: TrainedDQN = TrainedDQN(f'obs_{size}')

        # Initialize variables
        episode_step: int = 0

        terminated: bool = False
        truncated: bool = False
        not_finalized: bool = False

        path: list[Point] = [moving_point]

        observation, _ = env.init(moving_point, target, map)

        for step in range(max_steps):
            # Render
            if render:
                env.render()

            # Get action from policy
            action: int = agent.get_action(observation)

            # Step on the environment dynamics
            observation, terminated, truncated, info = env.step(action)

            # Update variables
            episode_step += 1
            path.append(path[-1] + info['next_position'])

            # Termination condition
            if terminated or truncated:
                break

            if step == max_steps-1:
                not_finalized = True

        return None if truncated or not_finalized else self._smooth_path(path)

    @staticmethod
    def _smooth_path(path: list[Point]) -> list[Point]:
        path_x: list[float] = [point.x for point in path]
        path_y: list[float] = [point.y for point in path]
        path_z: list[float] = [point.z for point in path]

        new_x: np.ndarray = savgol_filter(path_x, int(len(path_x)*0.9), 3)
        new_y: np.ndarray = savgol_filter(path_y, int(len(path_y)*0.9), 3)
        new_z: np.ndarray = savgol_filter(path_z, int(len(path_z)*0.9), 3)

        smooth_path: list[Point] = [Point(x, y, z) for x, y, z in zip(new_x, new_y, new_z)]

        return smooth_path

    @staticmethod
    def mock_path(step: float = 1.0, height: float = 1.10, distance: float = 0.5) -> list[Point]:
        path: list[Point] = []

        for i in range(220):
            x: float = distance
            y: float = np.sin(i*step/180.0*np.pi)*0.15 + 0.15
            z: float = height

            path.append(Point(x, y, z))

        return path

    @staticmethod
    def save_data(name: str, data: dict) -> None:
        data_frame: pd.DataFrame = pd.DataFrame(data)

        data_frame.to_csv(LOGGING_PATH+name+'.csv')
