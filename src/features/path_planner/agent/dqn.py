import numpy as np
import random

from keras.models import Sequential, load_model
from keras.layers import Dense
from collections import deque

from .interfaces import IAgent

MODEL_PATH = "src/features/path_planner/agent/models/"


class DQN(IAgent):
    REPLAY_MEMORY_SIZE: int = 10_000
    MIN_REPLAY_MEMORY_SIZE: int = 1024
    MINIBATCH_SIZE: int = 128

    UPDATE_TARGET_EVERY: int = 10

    EPSILON_DECAY: float = 0.9999
    MIN_EPSILON: float = 0.01

    DISCOUNT: float = 0.99

    HIDDEN_SIZE: int = 16

    def __init__(self, observation_size: int, action_size: int) -> None:
        # Network size
        self.observation_size: int = observation_size
        self.action_size: int = action_size

        # Models
        self.model: Sequential = self._create_model()
        self.target_model: Sequential = self._create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.model.trainable = True
        self.target_model.trainable = True

        # Replay memory
        self.replay_memory: deque = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        # Variables
        self.target_update_counter: int = 0
        self.epsilon: float = 0.99
        self.metrics: list[float] = [0.0, 0.0]

    def _create_model(self) -> Sequential:
        model: Sequential = Sequential()

        # Input layer and first hidden layer
        model.add(Dense(self.HIDDEN_SIZE,
                        input_dim=self.observation_size,
                        activation='relu'))

        # Hidden layer
        model.add(Dense(self.HIDDEN_SIZE,
                        activation='relu'))
        model.add(Dense(self.HIDDEN_SIZE,
                        activation='relu'))

        # Output layer
        model.add(Dense(self.action_size,
                        activation='linear'))

        # Compiler
        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    def get_action(self, observation: np.ndarray) -> int:
        action: int = np.random.randint(0, self.action_size)
        if np.random.random() > self.epsilon:
            prediction = self.model(np.array([observation]))
            action = np.argmax(prediction)  # type: ignore

        return action

    def train(self, observation: np.ndarray, new_observation: np.ndarray, action: int, reward: float, done: bool):
        self._update_memory((observation, new_observation, action, reward, done))
        self._fit_model()
        self._update_target_model(done)
        self._decay()

    def _update_memory(self, transition: tuple[np.ndarray, np.ndarray, int, float, bool]) -> None:
        self.replay_memory.append(transition)

    def _fit_model(self) -> None:
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
        observations: list[np.ndarray] = [transition[0] for transition in minibatch]
        new_observations: list[np.ndarray] = [transition[1] for transition in minibatch]

        qs: np.ndarray = np.array(self.model(np.array(observations)))
        new_qs: np.ndarray = np.array(self.target_model(np.array(new_observations)))

        input = []
        output = []
        for i, (observation, new_observation, action, reward, done) in enumerate(minibatch):
            new_q: float = reward if done else reward + self.DISCOUNT*np.max(new_qs[i])
            qs[i][action] = new_q

            input.append(observation)
            output.append(qs[i])

        self.metrics = self.model.train_on_batch(np.array(input), np.array(output))

    def _update_target_model(self, done: bool):
        if done:
            self.target_update_counter += 1

        if self.target_update_counter >= self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def _decay(self) -> None:
        if self.epsilon > self.MIN_EPSILON:
            self.epsilon *= self.EPSILON_DECAY
            self.epsilon = max(self.MIN_EPSILON, self.epsilon)

    def load_model(self, name: str) -> None:
        self.model = load_model(MODEL_PATH+name+'.h5')  # type:ignore

    def save_model(self, name: str) -> None:
        self.model.save(MODEL_PATH+name+'.h5')
