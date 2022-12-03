import bgplot as bgp

from ..entities import ManipulatorData
from ...core.entities import Point, Vector, Axes


SOF_AXES: Axes = Axes(Vector(1.0, 0.0, 0.0),
                      Vector(0.0, 1.0, 0.0),
                      Vector(0.0, 0.0, 1.0))

ORIGIN: Point = Point(0.0, 0.0, 0.0)
MANIPULATOR_ORIGIN: Point = Point(0.07, 0.13, 1.15)
MANIPULATOR_ORIGIN_PROJECTION: Point = Point(0.07, 0.13, 0.0)


class Graphics():
    def __init__(self) -> None:
        self.figure: bgp.Graphics = bgp.Graphics()
        self._set_figure_options()

    def _set_figure_options(self):
        self.figure.set_limits(xlim=(-0.5, 0.5),
                               ylim=(-0.5, 0.5),
                               zlim=(0.0, 1.5))

        self.figure.set_view(-15.0, 45.0)
        self.figure.disable('ticks', 'axes', 'walls')
        self.figure.set_background_color(bgp.Colors.white)

    def render(self, manipulator_data: ManipulatorData, virtual_point: Point, target: Point, path: list[Point]) -> None:
        # Manipulator
        self.figure.add_points(manipulator_data.positions, style='.-')
        self.figure.add_multiple_axes(manipulator_data.axes_list,
                                      manipulator_data.positions,
                                      length=0.025)

        # Virtual point
        self.figure.add_point(virtual_point, color=bgp.Colors.green)

        # Target
        self.figure.add_point(target, color=bgp.Colors.red)

        # Path
        self.figure.add_points(path, style='--', color=bgp.Colors.gray, linewidth=0.5)

        # Miscellaneous
        self.figure.add_points([MANIPULATOR_ORIGIN, MANIPULATOR_ORIGIN_PROJECTION],
                               style='.--', color=bgp.Colors.gray, linewidth=0.3)
        self.figure.add_axes(SOF_AXES, position=ORIGIN)

    def update(self, fps: int):
        self.figure.update(fps)

    def close(self) -> None:
        self.figure.close()