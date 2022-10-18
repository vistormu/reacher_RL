import numpy as np

from .....entities import ManipulatorData


INITIAL_CONFIGURATION: np.ndarray = np.array([[1.0, 0.0, 0.0, 0.07],
                                              [0.0, 0.7071, 0.7071, 0.13],
                                              [0.0, -0.7071, 0.7071, 1.15],
                                              [0.0, 0.0, 0.0, 1.0]])


class Kinematics:
    @classmethod
    def forward(cls, manipulator_data: ManipulatorData) -> list[np.ndarray]:
        d_values: list[float] = manipulator_data.d_values.copy()
        a_values: list[float] = manipulator_data.a_values.copy()
        alpha_values: list[float] = manipulator_data.alpha_values.copy()
        angles: list[float] = manipulator_data.angles.copy()

        transformation_matrix: np.ndarray = INITIAL_CONFIGURATION
        transformation_matrices: list[np.ndarray] = []

        for i in range(len(angles)):
            transformation_matrix = transformation_matrix @ \
                cls.translation(d_values[i], 'z') @ \
                cls.rotation(angles[i], 'z') @ \
                cls.translation(a_values[i], 'x') @ \
                cls.rotation(alpha_values[i], 'x')

            transformation_matrices.append(transformation_matrix)

        return transformation_matrices

    @staticmethod
    def rotation(angle: float, axis: str) -> np.ndarray:
        if axis == 'x':
            rotation_matrix = np.array([[1, 0, 0, 0],
                                        [0, np.cos(angle), -np.sin(angle), 0],
                                        [0, np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 0, 1]]).astype(float).round(5)
        elif axis == 'y':
            rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle), 0],
                                        [0, 1, 0, 0],
                                        [-np.sin(angle), 0, np.cos(angle), 0],
                                        [0, 0, 0, 1]]).astype(float).round(5)

        elif axis == 'z':
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                                        [np.sin(angle), np.cos(angle), 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]]).astype(float).round(5)
        else:
            raise Exception('Wrong axis input')

        return rotation_matrix

    @staticmethod
    def translation(value: float, axis: str) -> np.ndarray:
        if axis == 'x':
            translation_matrix = np.array([[1, 0, 0, value],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])
        elif axis == 'y':
            translation_matrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, value],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])
        elif axis == 'z':
            translation_matrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, value],
                                           [0, 0, 0, 1]])
        else:
            raise Exception('Wrong axis input')

        return translation_matrix
