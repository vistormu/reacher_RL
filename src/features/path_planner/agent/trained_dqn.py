import numpy as np
from keras.models import load_model

from .dqn import MODEL_PATH


class TrainedDQN:
    def __init__(self, name: str) -> None:
        self.model = load_model(MODEL_PATH+name+'.h5')

    def get_action(self, observation: np.ndarray) -> int:
        prediction = self.model(np.array([observation]))  # type: ignore
        return np.argmax(prediction)  # type: ignore
