from .implementations import GridWorldEnv
from .interfaces import IEnv


def make(id: str, *args, **kwargs) -> IEnv:
    if id == 'grid_world':
        return GridWorldEnv(*args, **kwargs)

    raise NameError('env id not found')
