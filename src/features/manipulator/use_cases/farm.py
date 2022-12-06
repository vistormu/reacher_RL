import numpy as np
from copy import deepcopy

from ..entities import ManipulatorData
from ...core.entities import OrientedPoint, Point, Vector, Axes


class Farm:
    @classmethod
    def iterate(cls, manipulator_data: ManipulatorData, target: OrientedPoint) -> ManipulatorData:
        backward_systems: list[OrientedPoint] = cls.backward(manipulator_data.systems,
                                                             target,
                                                             manipulator_data.dh_parameters.a,
                                                             manipulator_data.dh_parameters.d,
                                                             manipulator_data.dh_parameters.alpha,
                                                             manipulator_data.ik_parameters.phi,
                                                             manipulator_data.ik_parameters.mu,
                                                             )

        forward_systems, new_angles = cls.forward(backward_systems,
                                                  manipulator_data.systems[0],
                                                  manipulator_data.dh_parameters.a,
                                                  manipulator_data.dh_parameters.d,
                                                  manipulator_data.dh_parameters.alpha,
                                                  manipulator_data.ik_parameters.base_to,
                                                  )

        new_manipulator_data = deepcopy(manipulator_data)
        new_manipulator_data.systems = forward_systems
        new_manipulator_data.angles[1:] = new_angles

        return new_manipulator_data

    @staticmethod
    def backward(systems: list[OrientedPoint], target: OrientedPoint, a: list[float], d: list[float], alpha: list[float], phi: list[int], mu: list[int]) -> list[OrientedPoint]:
        # Initialize lists
        p: list[Point] = [system.position for system in systems]
        x: list[Vector] = [system.axes.x for system in systems]
        y: list[Vector] = [system.axes.y for system in systems]
        z: list[Vector] = [system.axes.z for system in systems]

        new_p: list[Point] = [systems[0].position] + [target.position]*(len(p)-1)
        new_x: list[Vector] = [systems[0].axes.x] + [target.axes.x]*(len(x)-1)
        new_y: list[Vector] = [systems[0].axes.y] + [target.axes.y]*(len(y)-1)
        new_z: list[Vector] = [systems[0].axes.z] + [target.axes.z]*(len(z)-1)

        for i in reversed(range(0, len(p)-1)):
            new_z[i] = new_y[i+1]*np.sin(alpha[i+1]) + new_z[i+1]*np.cos(alpha[i+1])
            new_p[i] = new_p[i+1] - Point(*new_x[i+1])*a[i+1] - Point(*new_y[i+1])*d[i+1]*np.sin(alpha[i+1]) - Point(*new_z[i+1])*d[i+1]*np.cos(alpha[i+1])

            j: int = i-2 if a[i]+d[i] == 0 else i-1

            v: np.ndarray = np.array(p[j] - new_p[i]) - np.dot(p[j] - new_p[i], new_z[i])*np.array(new_z[i])

            if a[i]+d[i] == 0:
                mu[i] = np.sign(np.dot(v, x[i]))

            new_x[i] = Vector(*((1-phi[i])*mu[i]*v + phi[i]*mu[i]*np.cross(v, new_z[i]))).normalize()
            new_y[i] = Vector(*((1-phi[i])*mu[i]*np.cross(new_z[i], v) + phi[i]*mu[i]*v)).normalize()

        return [OrientedPoint(p_i, Axes(x_i, y_i, z_i)) for (p_i, x_i, y_i, z_i) in zip(new_p, new_x, new_y, new_z)]

    @ staticmethod
    def forward(systems: list[OrientedPoint], base: OrientedPoint, a: list[float], d: list[float], alpha: list[float], base_to: int) -> tuple[list[OrientedPoint], list[float]]:
        # Initialize lists
        p: list[Point] = [oriented_point.position for oriented_point in systems]
        x: list[Vector] = [oriented_point.axes.x for oriented_point in systems]
        y: list[Vector] = [oriented_point.axes.y for oriented_point in systems]
        z: list[Vector] = [oriented_point.axes.z for oriented_point in systems]

        theta: list[float] = [0.0]*(len(p))

        new_p: list[Point] = [base.position]*len(p)
        new_x: list[Vector] = [base.axes.x]*len(x)
        new_y: list[Vector] = [base.axes.y]*len(y)
        new_z: list[Vector] = [base.axes.z]*len(z)

        for i in range(1, len(p)):
            u: np.ndarray = np.sign(np.dot(x[i], p[base_to] - new_p[i-1]))*np.array(p[base_to] - new_p[i-1]) if i == 1 and base_to != 0 else np.array(x[i])

            new_x[i] = Vector(*(u-np.dot(u, new_z[i-1])*new_z[i-1])).normalize()

            theta[i] = float(np.arctan2(np.dot(np.cross(new_x[i-1], new_x[i]), new_z[i-1]), np.dot(new_x[i], new_x[i-1])))

            new_p[i] = new_p[i-1] + Point(*new_x[i-1])*a[i]*np.cos(theta[i]) + Point(*new_y[i-1])*a[i]*np.sin(theta[i]) + Point(*new_z[i-1])*d[i]
            new_y[i] = (new_x[i-1]*(-np.cos(alpha[i])*np.sin(theta[i])) + new_y[i-1]*np.cos(alpha[i])*np.cos(theta[i]) + new_z[i-1]*np.sin(alpha[i])).normalize()
            new_z[i] = (new_x[i-1]*np.sin(alpha[i])*np.sin(theta[i]) + new_y[i-1]*(-np.sin(alpha[i])*np.cos(theta[i])) + new_z[i-1]*np.cos(alpha[i])).normalize()

        return [OrientedPoint(p_i, Axes(x_i, y_i, z_i)) for (p_i, x_i, y_i, z_i) in zip(new_p, new_x, new_y, new_z)], theta[1:]
