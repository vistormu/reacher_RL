from .implementations import GridWorldEnv, GridWorldEnvToDeploy
from .interfaces import IEnv


def get_env(id: str, *args, **kwargs) -> IEnv:
    if id == 'grid_world':
        return GridWorldEnv(*args, **kwargs)
    elif id == 'grid_world_to_deploy':
        return GridWorldEnvToDeploy(*args, **kwargs)

    raise NameError('env id not found')
