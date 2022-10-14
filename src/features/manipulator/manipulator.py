import yaml
from src.core.entities import ManipulatorData, Point, Vector, Axes

ASSETS_PATH: str = "src/assets/"


class Manipulator:
    @staticmethod
    def create(robot: str) -> ManipulatorData:
        path: str = f'{ASSETS_PATH}{robot}/dh.yaml'

        with open(path, "r") as file:
            try:
                dh_table = yaml.safe_load(file)
            except yaml.YAMLError as exception:
                print(exception)

        name: str = robot
        a_values: list[float] = dh_table[robot]['a']
        d_values: list[float] = dh_table[robot]['d']
        alpha_values: list[float] = dh_table[robot]['alpha']
        degrees_of_freedom: int = len(a_values)
        angles: list[float] = [0.0]*degrees_of_freedom
        positions: list[Point] = [Point(0.0, 0.0, 0.0)]*(degrees_of_freedom+1)
        x_vectors: list[Vector] = [
            Vector(0.0, 0.0, 0.0)]*(degrees_of_freedom+1)
        y_vectors: list[Vector] = [
            Vector(0.0, 0.0, 0.0)]*(degrees_of_freedom+1)
        z_vectors: list[Vector] = [
            Vector(0.0, 0.0, 0.0)]*(degrees_of_freedom+1)
        axes_list: list[Axes] = [
            Axes(*axes) for axes in zip(x_vectors, y_vectors, z_vectors)]

        manipulator_data: ManipulatorData = ManipulatorData(name=name,
                                                            a_values=a_values,
                                                            d_values=d_values,
                                                            alpha_values=alpha_values,
                                                            degrees_of_freedom=degrees_of_freedom,
                                                            positions=positions,
                                                            angles=angles,
                                                            axes_list=axes_list)

        return manipulator_data
