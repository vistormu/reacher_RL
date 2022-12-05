from .implementations import GridWorldEnv, GridWorldEnvToDeploy
from .interfaces import IEnv, IEnvToDeploy


def get_env(id: str, *args, **kwargs) -> IEnv:
    if id == 'grid_world':
        return GridWorldEnv(*args, **kwargs)

    raise NameError('env id not found')


def get_env_to_deploy(id: str, *args, **kwargs) -> IEnvToDeploy:
    if id == 'grid_world':
        return GridWorldEnvToDeploy(*args, **kwargs)

    raise NameError('env id not found')
