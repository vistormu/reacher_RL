from dataclasses import dataclass

from ...core.entities import OrientedPoint
from .dh_parameters import DHParameters
from .ik_parameters import IKParameters


@dataclass
class ManipulatorData:
    dh_parameters: DHParameters
    ik_parameters: IKParameters
    systems: list[OrientedPoint]
    angles: list[float]
