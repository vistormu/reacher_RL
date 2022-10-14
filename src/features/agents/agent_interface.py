from abc import ABC, abstractmethod


class IAgent(ABC):

    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def decay(self):
        pass