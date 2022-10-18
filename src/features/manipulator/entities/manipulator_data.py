from dataclasses import dataclass

from bgplot.entities import Point, Axes


@dataclass
class ManipulatorData:
    name: str
    a_values: list[float]
    d_values: list[float]
    alpha_values: list[float]
    degrees_of_freedom: int
    positions: list[Point]
    axes_list: list[Axes]
    angles: list[float]
