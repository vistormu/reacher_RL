import bgplot as bgp
import numpy as np

from typing import Optional
from matplotlib import cm
from matplotlib.colors import ListedColormap, Colormap
from bgplot.entities import Point as BgpPoint
from bgplot.entities import Line, Vector, Axes

from ..interfaces import IGraphics
from ..entities import OccupancyGrid, Point


class Graphics(IGraphics):
    def init(self, size: int) -> None:
        # Variables
        self.size: int = size

        # Figure
        self.figure: bgp.Graphics = bgp.Graphics()
        self._set_figure_options()
        self.colormap: ListedColormap = self._init_colormap()

    def _init_colormap(self) -> ListedColormap:
        colormap: Colormap = cm.get_cmap('Greys', 256)
        new_colors = colormap(np.linspace(0, 1, 256))

        transparent_color = np.array([0.0, 0.0, 0.0, 0.0])
        white = np.array([244/255, 243/255, 238/255, 1.0])
        new_colors[0, :] = transparent_color  # type:ignore
        new_colors[1:, :] = white*np.array(new_colors[1:, :])  # type:ignore

        # new_colors[:, -1] = 0.2  # type:ignore

        new_colormap = ListedColormap(new_colors)  # type:ignore

        return new_colormap

    def _set_figure_options(self):
        self.figure.set_limits(xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0))  # type: ignore
        self.figure.set_view(-15.0, 45.0)
        self.figure.disable('ticks', 'axes', 'walls')
        self.figure.set_background_color(bgp.Colors.white)

    def render(self, moving_point: Point, target: Point, map: Optional[OccupancyGrid]) -> None:
        # Occupancy grid
        if map is not None:
            lineal_x: np.ndarray = np.linspace(0.0, 1.0, map.size, endpoint=False)
            lineal_y: np.ndarray = np.linspace(0.0, 1.0, map.size, endpoint=False)
            surface_x, surface_y = np.meshgrid(lineal_x, lineal_y)
            self.figure._ax.plot_surface(surface_x, surface_y, map.map.T/self.size, cmap=self.colormap, linewidth=0, antialiased=False, zorder=0)  # type:ignore

        # Moving point
        bgp_moving_point: BgpPoint = BgpPoint(moving_point.x/self.size, moving_point.y/self.size, moving_point.z/self.size)
        self.figure.add_point(bgp_moving_point, color=bgp.Colors.green)

        # Target
        bgp_target: BgpPoint = BgpPoint(target.x/self.size, target.y/self.size, target.z/self.size)
        self.figure.add_point(bgp_target, color=bgp.Colors.red)

        # Line
        line: Line = Line.from_two_points(bgp_moving_point, bgp_target)
        self.figure.add_line(line, style='--', color=bgp.Colors.gray, linewidth=0.2)

        # Axes
        self.figure.add_axes(Axes(Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0), Vector(0.0, 0.0, 1.0)), length=0.05)

    def update(self, fps: int) -> None:
        self.figure.update(fps)

    def close(self) -> None:
        self.figure.close()

    def set_title(self, title: str) -> None:
        self.figure.set_title(title)
