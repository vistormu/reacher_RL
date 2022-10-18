from tqdm import tqdm

from ...core import Logger
from .agents import get_agent
from .envs import get_env


class Training:
    def train(self):
        # Agent and env
        env = get_env('grid_world', size=0.025)
        agent = get_agent('deep_q_agent', size=(env.observation_space_size,
                                                64,
                                                env.action_space_size))
        agent.load_model()  # Si no hay no funciona

        # Training constants
        EPISODES: int = 100
        MAX_STEPS: int = 200
        SHOW_AFTER: int = 90

        # Variables
        times_completed: int = 0

        # Train model
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

        Logger.info(f'times completed: {times_completed}/{EPISODES} ({times_completed*100//EPISODES}%)')
