from dataclasses import dataclass


@dataclass
class IKParameters:
    phi: list[int]
    mu: list[int]
    base_to: int
    