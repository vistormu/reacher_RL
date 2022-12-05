import numpy as np
from keras import models

from .deep_q_agent import MODEL_PATH

from ...interfaces import ITrainedAgent


class TrainedDeepQAgent(ITrainedAgent):
    def __init__(self) -> None:
        self.model = models.load_model(MODEL_PATH)

    def get_action(self, observation: np.ndarray) -> int:
        prediction = self.model(np.array([observation]))  # type: ignore
        action: int = np.argmax(prediction)  # type: ignore

        return action
