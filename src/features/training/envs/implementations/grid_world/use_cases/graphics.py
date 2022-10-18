import bgplot as bgp
from bgplot.entities import Point, Line, Vector, Plane

from ....interfaces import IGraphics


class Graphics(IGraphics):
    def init(self, limits: list[tuple[float, float]]) -> None:
        self.figure: bgp.Graphics = bgp.Graphics()
        self.limits: list[tuple[float, float]] = limits
        self._set_figure_options(limits)

    def _set_figure_options(self, limits: list[tuple[float, float]]):
        self.figure.set_limits(xlim=limits[0],
                               ylim=limits[1],
                               zlim=limits[2])

        self.figure.set_view(-15.0, 45.0)
        self.figure.disable('ticks', 'axes', 'walls')
        self.figure.set_background_color(bgp.Colors.white)

    def render(self, virtual_point: Point, target: Point) -> None:
        virtual_x: Point = Point(self.limits[0][0], virtual_point.y, virtual_point.z)
        virtual_y: Point = Point(virtual_point.x, self.limits[1][0], virtual_point.z)
        virtual_z: Point = Point(virtual_point.x, virtual_point.y, self.limits[2][0])

        target_x: Point = Point(self.limits[0][0], target.y, target.z)
        target_y: Point = Point(target.x, self.limits[1][0], target.z)
        target_z: Point = Point(target.x, target.y, self.limits[2][0])

        self.figure.add_points([virtual_x, target_x], style='--', linewidth=0.2)
        self.figure.add_points([virtual_y, target_y], style='--', linewidth=0.2)
        self.figure.add_points([virtual_z, target_z], style='--', linewidth=0.2)

        self.figure.add_point(virtual_point, color=bgp.Colors.green)
        self.figure.add_point(target, color=bgp.Colors.red)

    def update(self, fps: int) -> None:
        self.figure.update(fps)

    def close(self) -> None:
        self.figure.close()

    def set_title(self, title: str) -> None:
        self.figure.set_title(title)
