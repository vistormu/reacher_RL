import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter

from ...core import Logger
from ..core.entities import Point
from .agents import get_agent
from .envs import get_env


class PathPlanner:
    def __init__(self, env_id: str, agent_id: str) -> None:
        self.env_id: str = env_id
        self.agent_id: str = agent_id

    @staticmethod
    def mock_path(step: float = 1.0, height: float = 1.10, distance: float = 0.5) -> list[Point]:
        path: list[Point] = []

        for i in range(220):
            x: float = distance
            y: float = np.sin(i*step/180.0*np.pi)*0.15 + 0.15
            z: float = height

            path.append(Point(x, y, z))

        return path

    def train(self):
        # Agent and env
        env = get_env(self.env_id, size=0.025)
        agent = get_agent(self.agent_id, size=(env.observation_space_size,
                                               64,
                                               env.action_space_size))

        try:
            agent.load_model()
            Logger.info('model loaded')
        except:
            pass

        # Training constants
        EPISODES: int = 100
        MAX_STEPS: int = 200
        SHOW_AFTER: int = 95

        # Variables
        times_completed: int = 0

        # Train model
        Logger.info('training model')
        for episode in tqdm(range(1, EPISODES+1)):
            # Varibable resetting
            done: bool = False
            episode_step: int = 0
            observation = env.reset()

            if episode == SHOW_AFTER:
                input('press any key to continue...')

            while not done:
                # Check max steps in episode
                if episode_step >= MAX_STEPS:
                    break

                # Render
                if episode >= SHOW_AFTER:
                    env.render()

                # Get action from policy
                action: int = agent.get_action(observation)

                # Step on the environment dynamics
                new_observation, reward, done, _ = env.step(action)

                # Train agent
                agent.train(observation, new_observation, action, reward, done)

                # Update variables
                observation = new_observation
                episode_step += 1

                if done:
                    times_completed += 1

            agent.decay()

        env.close()
        agent.save_model()
        Logger.info('model saved')

        Logger.info(f'times completed: {times_completed}/{EPISODES} ({times_completed*100//EPISODES}%)')

    def get_path(self, virtual_point: Point, target: Point):
        # Env and agent
        env = get_env(self.env_id + '_to_deploy', size=0.01)
        agent = get_agent('trained_' + self.agent_id)

        # Constants
        MAX_STEPS: int = 200

        # Initialize variables
        path: list[Point] = [virtual_point]
        done: bool = False
        observation: np.ndarray = env.init(virtual_point,
                                           target)
        episode_step: int = 0

        while not done:
            # Check max steps in episode
            if episode_step >= MAX_STEPS:
                break

            # Get action from policy
            action: int = agent.get_action(observation)

            # Step on the environment dynamics
            new_observation, done, info = env.step(action)

            # Update variables
            observation = new_observation
            episode_step += 1
            path.append(info['virtual_point'])

        return self._smooth_path(path)

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
