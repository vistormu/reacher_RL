import numpy as np

from perlin_noise import PerlinNoise


class OccupancyGrid:
    def __init__(self, size: int, seed: int) -> None:
        self.map: np.ndarray = self._create_occupancy_map(size, seed)
        self.size: int = size

    def __repr__(self) -> str:
        return f'{self.map}'

    def _create_occupancy_map(self, size: int, seed: int) -> np.ndarray:
        noise: PerlinNoise = PerlinNoise(octaves=1, seed=seed)

        occupancy_map = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                noise_value: float = noise([i/size, j/size])
                occupancy_map[i, j] = self._sigmoid(noise_value, m=20.0, n=0.0, k=0.9)*size

        return occupancy_map.astype(int)

    @staticmethod
    def _sigmoid(x: float, m: float, n: float = 0.0, k: float = 1.0) -> float:
        return k/(1+np.exp(-m*(x-n)))

    def get_min_value(self) -> int:
        return np.min(self.map)

    def get_max_value(self) -> int:
        return np.max(self.map)

    def get_value(self, x: int, y: int) -> int:
        return self.map[x, y]

    def get_min_index(self) -> tuple[int, int]:
        return divmod(self.map.argmin(), self.map.shape[1])  # type:ignore

    def get_max_index(self) -> tuple[int, int]:
        return divmod(self.map.argmax(), self.map.shape[1])  # type:ignore
