from .implementations import MockRepository, MujocoRepository

from ..repository import ManipulatorRepository


def get_repository(id: str, *args, **kwargs) -> ManipulatorRepository:
    if id == 'mock':
        return MockRepository()
    if id == 'mujoco':
        return MujocoRepository(*args, **kwargs)

    raise NameError('repository id not found')
