import bgplot as bgp

from ..entities import ManipulatorData
from ...core.entities import Point


class Graphics():
    def __init__(self) -> None:
        self.figure: bgp.Graphics = bgp.Graphics()
        self._set_figure_options()

    def _set_figure_options(self):
        self.figure.set_limits(xlim=(-0.1, 0.5),  # type:ignore
                               ylim=(-0.1, 0.5),  # type:ignore
                               zlim=(0.0, 0.5))  # type:ignore

        self.figure.set_view(-15.0, 45.0)
        self.figure.disable('ticks', 'axes', 'walls')
        self.figure.set_background_color(bgp.Colors.white)

    def render(self, manipulator_data: ManipulatorData, virtual_point: Point, target: Point, path: list[Point]) -> None:
        # Manipulator
        self.figure.add_oriented_points(manipulator_data.systems, style='.-', length=0.025)  # type:ignore

        # Virtual point
        self.figure.add_point(virtual_point, color=bgp.Colors.green)

        # Target
        self.figure.add_point(target, color=bgp.Colors.red)

        # Path
        self.figure.add_points(path, style='--', color=bgp.Colors.gray, linewidth=0.5)

    def update(self, fps: int):
        self.figure.update(fps)

    def close(self) -> None:
        self.figure.close()
