from .implementations import DeepQAgent, TrainedDeepQAgent
from .agent_interface import IAgent


def get_agent(agent_id: str, *args, **kwargs) -> IAgent:
    if agent_id == 'deep_q_agent':
        return DeepQAgent(*args, **kwargs)
    elif agent_id == 'trained_deep_q_agent':
        return TrainedDeepQAgent()

    raise NameError('agent id not found')
