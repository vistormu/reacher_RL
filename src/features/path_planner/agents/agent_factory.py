from .implementations import DeepQAgent, TrainedDeepQAgent
from .interfaces import IAgent, ITrainedAgent


def get_agent(agent_id: str, *args, **kwargs) -> IAgent:
    if agent_id == 'deep_q_agent':
        return DeepQAgent(*args, **kwargs)

    raise NameError('agent id not found')


def get_trained_agent(id: str) -> ITrainedAgent:
    if id == 'deep_q_agent':
        return TrainedDeepQAgent()

    raise NameError('agent id not found')
