import numpy as np

from bgplot.entities import Point, Vector, Axes
from bgplot.entities import OrientedPoint as BgpOrientedPoint


class OrientedPoint(BgpOrientedPoint):

    def to_htm(self) -> np.ndarray:
        """
        transforms an oriented point instance to the homogeneous transformation matrix representation

        Returns
        -------
        out : np.ndarray
            the homogeneous transformation matrix as a np array
        """
        return np.array([[*self.axes.x, 0],
                         [*self.axes.y, 0],
                         [*self.axes.z, 0],
                         [*self.position, 1]]).T

    @classmethod
    def from_htm(cls, transformation_matrix: np.ndarray):
        point: Point = Point(*transformation_matrix[0:3, 3])
        x: Vector = Vector(*transformation_matrix[0:3, 0])
        y: Vector = Vector(*transformation_matrix[0:3, 1])
        z: Vector = Vector(*transformation_matrix[0:3, 2])

        return cls(point, Axes(x, y, z))
