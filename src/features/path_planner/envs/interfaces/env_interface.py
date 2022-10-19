import gym


class IEnv(gym.Env):
    def __init__(self) -> None:
        self.action_space_size: int
        self.observation_space_size: int
