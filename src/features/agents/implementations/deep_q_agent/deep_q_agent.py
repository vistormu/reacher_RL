import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from collections import deque

from ...agent_interface import IAgent

MODEL_PATH = "src/features/agents/implementations/deep_q_agent/model/"


class DeepQAgent(IAgent):

    REPLAY_MEMORY_SIZE: int = 50_000
    MIN_REPLAY_MEMORY_SIZE: int = 1000
    MINIBATCH_SIZE: int = 64

    UPDATE_TARGET_EVERY: int = 5

    EPSILON_DECAY: float = 0.99975
    MIN_EPSILON: float = 0.001

    DISCOUNT: float = 0.99

    def __init__(self, size: tuple[int, int, int]) -> None:
        self.input_size, self.hidden_size, self.output_size = size

        # Models
        self.model: Sequential = self._create_model()
        self.target_model: Sequential = self._create_model()
        self.target_model.set_weights(self.model.get_weights())

        # Variables
        self.replay_memory: deque = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.target_update_counter: int = 0
        self.epsilon: float = 1.0

    def get_action(self, state: np.ndarray) -> int:
        action: int = np.random.randint(0, self.output_size)
        if np.random.random() > self.epsilon:
            prediction = self.model(np.array([state]))
            action = np.argmax(prediction)

        return action

    def train(self, observation: np.ndarray, new_observation: np.ndarray, action: int, reward: int, done: bool) -> None:
        self._update_memory(observation, new_observation, action, reward, done)
        self._train_model()

        # Update target network counter every episode
        if done:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def decay(self) -> None:
        if self.epsilon > self.MIN_EPSILON:
            self.epsilon *= self.EPSILON_DECAY
            self.epsilon = max(self.MIN_EPSILON, self.epsilon)

    def save_model(self):
        self.model.save(MODEL_PATH)

    def load_model(self):
        self.model.save(MODEL_PATH)

    def _create_model(self) -> Sequential:
        model: Sequential = Sequential()

        # Input layer and first hidden layer
        model.add(Dense(self.hidden_size,
                        input_dim=self.input_size,
                        activation='relu'))

        # Hidden layer
        model.add(Dense(self.hidden_size,
                        activation='relu'))

        # Output layer
        model.add(Dense(self.output_size,
                        activation='linear'))

        # Compiler
        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    def _update_memory(self, observation: np.ndarray, new_observation: np.ndarray, action: int, reward: int, done: bool) -> None:
        transition: tuple[np.ndarray, np.ndarray, int, int, bool] = (
            observation, new_observation, action, reward, done)
        self.replay_memory.append(transition)

    def _train_model(self) -> None:
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get minibatch from memory
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        observations, new_observations, actions, rewards, dones = zip(
            *minibatch)

        current_qs = self.model(np.array(observations))
        future_qs = self.model(np.array(new_observations))

        current_qs = np.array(current_qs)
        future_qs = np.array(future_qs)

        X = []
        y = []

        for index, (observation, new_state, action, reward, done) in enumerate(minibatch):

            if not done:
                max_future_q = np.max(future_qs[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                # new_q = reward
                new_q = 0

            # Update Q value for given state
            current_q = current_qs[index]
            current_q[action] = new_q

            # And append to our training data
            X.append(observation)
            y.append(current_q)

        # Fit on all samples as one batch
        self.model.fit(np.array(X),
                       np.array(y),
                       batch_size=self.MINIBATCH_SIZE,
                       verbose=0,
                       shuffle=False)
