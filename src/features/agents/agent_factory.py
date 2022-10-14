from .implementations import DeepQAgent
from .agent_interface import IAgent


def get(agent_type: str, *args, **kwargs) -> IAgent:
    if agent_type == 'deep_q_agent':
        return DeepQAgent(*args, **kwargs)

    raise NameError('agent id not found')
