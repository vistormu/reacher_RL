from .implementations import MockManipulatorRepository, MujocoRepository

from ..repository import ManipulatorRepository


def get_repository(id: str, *args, **kwargs) -> ManipulatorRepository:
    if id == 'mock':
        return MockManipulatorRepository()
    if id == 'mujoco':
        return MujocoRepository(*args, **kwargs)

    raise NameError('repository id not found')
