from typing import NamedTuple


class Point(NamedTuple):
    x: int
    y: int
    z: int

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.x}, {self.y}, {self.z})'

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar: int):
        return Point(self.x*scalar, self.y*scalar, self.z*scalar)

    def __rmul__(self, scalar: int):
        return Point(self.x*scalar, self.y*scalar, self.z*scalar)
