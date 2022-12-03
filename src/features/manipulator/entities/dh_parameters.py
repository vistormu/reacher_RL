from dataclasses import dataclass


@dataclass
class DHParameters:
    a: list[float]
    d: list[float]
    alpha: list[float]
