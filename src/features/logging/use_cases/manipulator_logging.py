import csv
from pathlib import Path

from ...core.entities import Point


class ManipulatorLogging:
    def __init__(self, logging_path: str) -> None:
        self.angles_filename: str = logging_path + 'angles.csv'
        self.positions_filename: str = logging_path + 'positions.csv'

        self._unlink_file(self.positions_filename)
        self._unlink_file(self.angles_filename)

    @staticmethod
    def _unlink_file(filename: str) -> None:
        if Path(filename).is_file():
            Path(filename).unlink()

    def angles_to_csv(self, angles: list[float], digits: int = 2) -> None:
        angles_to_log: list[float] = [round(angle, digits) for angle in angles]

        with open(self.angles_filename, 'a+') as file:
            writer = csv.writer(file)
            writer.writerow(angles_to_log)

    def positions_to_csv(self, positions: list[Point], digits: int = 2) -> None:
        positions_to_log: list[list[float]] = []
        for position in positions:
            rounded_position: list[float] = [round(value, digits) for value in position]
            positions_to_log.append(rounded_position)

        with open(self.positions_filename, 'a+') as file:
            writer = csv.writer(file)
            writer.writerow(positions_to_log)
