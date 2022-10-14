import numpy as np
from bgplot.entities import Point, Axes, Vector
from copy import deepcopy

from src.core.entities import ManipulatorData


class FabrikVanilla:
    @classmethod
    def iterate(cls, manipulator_data: ManipulatorData, target: Point, target_axes: Axes) -> ManipulatorData:
        # Variables
        a: list[float] = manipulator_data.a_values.copy()
        d: list[float] = manipulator_data.d_values.copy()
        alpha: list[float] = manipulator_data.alpha_values.copy()

        positions: list[Point] = manipulator_data.positions.copy()
        axes_list: list[Axes] = manipulator_data.axes_list.copy()

        base: Point = manipulator_data.positions[0]
        base_axes: Axes = manipulator_data.axes_list[0]

        # Perform backward and forward propagation
        backward_positions, backward_axes_list = cls._backward(positions=positions,
                                                               axes_list=axes_list,
                                                               target=target,
                                                               target_axes=target_axes,
                                                               a=a,
                                                               d=d,
                                                               alpha=alpha)

        forward_positions, forward_axes_list, angles = cls._forward(positions=backward_positions,
                                                                    axes_list=backward_axes_list,
                                                                    target=base,
                                                                    target_axes=base_axes,
                                                                    a=a,
                                                                    d=d,
                                                                    alpha=alpha)

        new_manipulator_data: ManipulatorData = deepcopy(manipulator_data)
        new_manipulator_data.positions = forward_positions
        new_manipulator_data.axes_list = forward_axes_list
        new_manipulator_data.angles = angles

        return new_manipulator_data

    @staticmethod
    def _backward(positions: list[Point], axes_list: list[Axes], target: Point, target_axes: Axes, a: list[float], d: list[float], alpha: list[float]) -> tuple[list[Point], list[Axes]]:
        # 1. Place end effector on the target's position and orientation
        new_positions: list[Point] = [target]
        new_axes_list: list[Axes] = [target_axes]
        positions.pop()
        axes_list.pop()

        for i in range(len(positions)-1):
            # 2. Calculate z_i and p_i
            last_axes: np.ndarray = np.array([*new_axes_list[-1]]).T
            z_transformation: np.ndarray = np.array([0,
                                                    np.sin(alpha[-1-i]),
                                                    np.cos(alpha[-1-i])])
            p_transformation: np.ndarray = np.array([-a[-1-i],
                                                    -d[-1-i] *
                                                    np.sin(alpha[-1-i]),
                                                    -d[-1-i]*np.cos(alpha[-1-i])]).T

            z_i: Vector = Vector(*(last_axes @ z_transformation)).normalize()
            p_i: Point = new_positions[-1] + \
                Point(*(last_axes @ p_transformation))

            positions.pop()
            axes_list.pop()

            # 3. projection of the vector
            u_vector: Vector = Vector(*(positions[-1]-p_i)).normalize()
            u: np.ndarray = np.array(u_vector)
            n: np.ndarray = np.array(z_i)
            v: np.ndarray = u - np.dot(u, n)*n

            # 4. Get v with the proper direction
            t: np.ndarray = np.array([-a[-2-i],
                                      -d[-2-i]*np.sin(alpha[-2-i]),
                                      -d[-2-i]*np.cos(alpha[-2-i])]).T
            next_vector: Vector = Vector(
                *np.array([v, v, v]).T @ t).normalize()

            # 5. Calculate x_i and y_i
            if a[-2-i]:
                x_i: Vector = Vector(*next_vector)
                y_i: Vector = Vector(*(np.cross(z_i, x_i)))
            else:
                y_i: Vector = Vector(*next_vector)
                x_i: Vector = Vector(*(np.cross(y_i, z_i)))

            axes_i: Axes = Axes(x_i, y_i, z_i)
            new_axes_list.append(axes_i)
            new_positions.append(p_i)

        return list(reversed(new_positions)), list(reversed(new_axes_list))

    @ staticmethod
    def _forward(positions: list[Point], axes_list: list[Axes], target: Point, target_axes: Axes, a: list[float], d: list[float], alpha: list[float]) -> tuple[list[Point], list[Axes], list[float]]:
        # 1. Place first point back at the base
        new_positions: list[Point] = [target]
        new_axes_list: list[Axes] = [target_axes]
        new_angles: list[float] = []

        for i in range(len(positions)):
            # Choose next point to join
            if d[i] != 0.0:
                next_p_list: list[int] = list(np.nonzero(d[i+1:])[0])
            elif a[i] != 0.0:
                next_p_list: list[int] = list(np.nonzero(a[i+1:])[0])

            if not next_p_list:
                next_p_index = 0
            else:
                next_p_index = next_p_list[0]

            # 2. Calculate x_i
            if next_p_index == 0 and d[i] != 0:
                u: np.ndarray = np.array(axes_list[0].x)
            else:
                u_vector: Vector = Vector(
                    *(new_positions[i]-positions[next_p_index])).normalize()
                u: np.ndarray = np.array(u_vector)

            n: np.ndarray = np.array(new_axes_list[i].z)
            v: np.ndarray = u - np.dot(u, n)*n

            x_i: Vector = Vector(*v)

            # Calculate angle
            v1: np.ndarray = np.array(x_i.normalize())
            v2: np.ndarray = np.array(new_axes_list[i].x.normalize())
            n: np.ndarray = np.array(new_axes_list[i].z.normalize())

            def get_angle(v1, v2, n) -> float:
                return np.arctan2(
                    np.dot(np.cross(v2, v1), n), np.dot(v1, v2))

            angle: float = get_angle(v1, v2, n)

            # if angle < 0.0:
            #     angle += 2.0*np.pi

            new_angles.append(angle)

            # Calculate p_i, y_i and z_i
            x: np.ndarray = np.array(new_axes_list[i].x)
            y: np.ndarray = np.array(new_axes_list[i].y)
            z: np.ndarray = np.array(new_axes_list[i].z)
            p: np.ndarray = np.array(new_positions[i])

            new_x: np.ndarray = np.dot(np.array([x, y, z]).T,
                                       [np.cos(angle), np.sin(angle), 0])

            new_y: np.ndarray = np.dot(np.array([x, y, z]).T,
                                       [-np.cos(alpha[i])*np.sin(angle), np.cos(alpha[i])*np.cos(angle), np.sin(alpha[i])])

            new_z: np.ndarray = np.dot(np.array([x, y, z]).T,
                                       [np.sin(alpha[i])*np.sin(angle), -np.sin(alpha[i])*np.cos(angle), np.cos(alpha[i])])

            new_p: np.ndarray = p + np.dot(np.array([x, y, z]).T,
                                           [a[i]*np.cos(angle), a[i]*np.sin(angle), d[i]])

            p_i: Point = Point(*new_p)
            x_i: Vector = Vector(*new_x).normalize()
            y_i: Vector = Vector(*new_y).normalize()
            z_i: Vector = Vector(*new_z).normalize()
            axes_i: Axes = Axes(x_i, y_i, z_i)

            new_positions.append(p_i)
            new_axes_list.append(axes_i)
            positions.pop(0)
            axes_list.pop(0)

        return new_positions, new_axes_list, new_angles