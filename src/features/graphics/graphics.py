import bgplot as bgp

from src.core.entities import Point, Axes, Vector, Line, ManipulatorData


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

    def show_initial_configuration(self, manipulator_data: ManipulatorData, target: Point, target_axes: Axes) -> None:
        # Manipulator
        self.figure.add_oriented_points(manipulator_data.positions,
                                        manipulator_data.axes_list,
                                        style='.-', length=0.025)

        # Target
        self.figure.add_oriented_point(target,
                                       target_axes,
                                       color=bgp.Colors.red)

        # Miscellaneous
        self.figure.add_points([MANIPULATOR_ORIGIN, MANIPULATOR_ORIGIN_PROJECTION],
                               style='.--', color=bgp.Colors.gray)
        self.figure.add_axes(SOF_AXES, position=ORIGIN)

        # Graphics
        self.figure.set_title('initial configuration')
        self.figure.show()

    def show_current_state(self, manipulator_data: ManipulatorData, target: Point, target_axes: Axes) -> None:
        # Manipulator
        self.figure.add_oriented_points(manipulator_data.positions,
                                        manipulator_data.axes_list,
                                        style='.-', length=0.025)

        # Target
        self.figure.add_oriented_point(target,
                                       target_axes,
                                       color=bgp.Colors.red)

        # Miscellaneous
        self.figure.add_points([MANIPULATOR_ORIGIN, MANIPULATOR_ORIGIN_PROJECTION],
                               style='.--', color=bgp.Colors.gray)
        self.figure.add_axes(SOF_AXES, position=ORIGIN)

        # Graphics
        self.figure.set_title('current state')
        self.figure.update(30)

    def close(self):
        self.figure.close()
