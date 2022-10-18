from .implementations import GridWorldEnv
from .interfaces import IEnv


def get_env(id: str, *args, **kwargs) -> IEnv:
    if id == 'grid_world':
        return GridWorldEnv(*args, **kwargs)

    raise NameError('env id not found')
