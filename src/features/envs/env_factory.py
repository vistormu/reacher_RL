from .implementations import GridWorldEnv
from .interfaces import IEnv


def make(id: str) -> IEnv:
    if id == 'grid_world':
        return GridWorldEnv()

    raise NameError('env id not found')