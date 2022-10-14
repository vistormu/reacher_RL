import numpy as np

from src.core.entities import ManipulatorData, Point, Vector, Axes


INITIAL_CONFIGURATION: np.ndarray = np.array([[1.0, 0.0, 0.0, 0.07],
                                              [0.0, 0.7071, 0.7071, 0.13],
                                              [0.0, -0.7071, 0.7071, 1.15],
                                              [0.0, 0.0, 0.0, 1.0]])


class ManipulatorDataGetter:
    @classmethod
    def get(cls, transformation_matrices: list[np.ndarray], manipulator_data: ManipulatorData) -> ManipulatorData:
        positions: list[Point] = cls.get_positions(transformation_matrices)
        x_vectors: list[Vector] = cls.get_x_vectors(transformation_matrices)
        y_vectors: list[Vector] = cls.get_y_vectors(transformation_matrices)
        z_vectors: list[Vector] = cls.get_z_vectors(transformation_matrices)
        axes_list: list[Axes] = [
            Axes(*axes) for axes in zip(x_vectors, y_vectors, z_vectors)]

        manipulator_data.positions = positions
        manipulator_data.axes_list = axes_list

        return manipulator_data

    @staticmethod
    def get_positions(transformation_matrices: list[np.ndarray]) -> list[Point]:
        positions: list[Point] = [Point(*transformation_matrix[:3, -1])
                                  for transformation_matrix in transformation_matrices]

        # INITIAL CONFIGURATION
        positions.insert(0, Point(*INITIAL_CONFIGURATION[0:3, -1]))

        return positions

    @staticmethod
    def get_x_vectors(transformation_matrices: list[np.ndarray]) -> list[Vector]:
        x_vectors: list[Vector] = [Vector(*transformation_matrix[:3, 0]).normalize()
                                   for transformation_matrix in transformation_matrices]

        x_vectors.insert(0, Vector(*INITIAL_CONFIGURATION[0:3, 0]))

        return x_vectors

    @staticmethod
    def get_y_vectors(transformation_matrices: list[np.ndarray]) -> list[Vector]:
        y_vectors: list[Vector] = [Vector(*transformation_matrix[:3, 1]).normalize()
                                   for transformation_matrix in transformation_matrices]

        y_vectors.insert(0, Vector(*INITIAL_CONFIGURATION[0:3, 1]))

        return y_vectors

    @staticmethod
    def get_z_vectors(transformation_matrices: list[np.ndarray]) -> list[Vector]:
        z_vectors: list[Vector] = [Vector(*transformation_matrix[:3, 2]).normalize()
                                   for transformation_matrix in transformation_matrices]

        z_vectors.insert(0, Vector(*INITIAL_CONFIGURATION[0:3, 2]))

        return z_vectors
