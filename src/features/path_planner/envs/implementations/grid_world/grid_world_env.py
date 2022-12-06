import numpy as np

from ...interfaces import IEnv
from .use_cases import Observer, Translator, Graphics
from .....core.entities import Point, Vector

RENDER_FPS: int = 120


class GridWorldEnv(IEnv):
    def __init__(self, size: float) -> None:
        # Env variables
        self.size: float = size
        self.limits: list[tuple[float, float]] = [(-1.0, 1.0),
                                                  (-1.0, 1.0),
                                                  (-1.0, 1.0)]

        # Entities
        self.target: Point = Point(0.0, 0.0, 0.0)
        self.virtual_point: Point = Point(0.0, 0.0, 0.0)

        # Translator and observer
        self.translator: Translator = Translator()
        self.observer: Observer = Observer(self.size, self.limits)

        # Graphics
        self.graphics: Graphics = Graphics()

        # Action space
        self.action_space_size: int = self.translator.action_space_size

        # Observation space
        self.observation_space_size: int = self.observer.observation_space_size

        # Other variables
        self.window: bool = False
        self.episode_step: int = 0

    def reset(self) -> tuple[np.ndarray, dict]:
        # Reset episode step
        self.episode_step = 0

        # Set target to a new position
        target_x: float = np.random.uniform(*self.limits[0])
        target_y: float = np.random.uniform(*self.limits[1])
        target_z: float = np.random.uniform(*self.limits[2])

        self.target = Point(target_x, target_y, target_z)

        # Get observation and info
        observation: np.ndarray = self.observer.get_observation(self.virtual_point,
                                                                self.target)
        info: dict = self.observer.get_info(self.virtual_point, self.target)

        return (observation, info)

    def step(self, action: int) -> tuple[np.ndarray, int, bool, dict]:
        # Increase episode step
        self.episode_step += 1

        # Move the virtual point from the action
        direction: Vector = self.translator.get_direction(action, self.size)
        new_position: Point = self.virtual_point + Point(*direction)

        self.virtual_point = Point(np.clip(new_position.x, *self.limits[0]),
                                   np.clip(new_position.y, *self.limits[1]),
                                   np.clip(new_position.z, *self.limits[2]))

        # Get observation, reward, done and info
        observation = self.observer.get_observation(self.virtual_point,
                                                    self.target)
        reward = self.observer.get_reward(self.virtual_point,
                                          self.target)
        done = self.observer.is_done(self.virtual_point,
                                     self.target)
        info = self.observer.get_info(self.virtual_point, self.target)

        return observation, reward, done, info

    def render(self) -> None:
        if not self.window:
            self.graphics.init(self.limits)
            self.window = True

        self.graphics.render(self.virtual_point,
                             self.target)
        self.graphics.set_title(f'{self.episode_step}')
        self.graphics.update(RENDER_FPS)

    def close(self):
        self.graphics.close()
