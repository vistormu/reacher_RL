import csv
from pathlib import Path
from src.core.entities import Point

LOGGING_PATH = 'src/features/logging/results/'
ANGLES_FILENAME = 'angles.csv'
PATH_FILENAME = 'position.csv'
TARGET_PATH_FILENAME = 'target.csv'


class Logging:
    def __init__(self) -> None:
        Path(LOGGING_PATH).mkdir(parents=True, exist_ok=True)

        self.angles_filename: str = LOGGING_PATH + 'angles.csv'
        self._unlink_file(self.angles_filename)

        self.position_filename: str = LOGGING_PATH + 'position.csv'
        self._unlink_file(self.position_filename)

        self.target_filename: str = LOGGING_PATH + 'target.csv'
        self._unlink_file(self.target_filename)

    @staticmethod
    def _unlink_file(filename: str):
        if Path(filename).is_file():
            Path(filename).unlink()

    def angles_to_csv(self, angles: list[float], digits: int = 2) -> None:
        angles_to_log: list[float] = [round(angle, digits) for angle in angles]
        with open(self.angles_filename, 'a+') as file:
            writer = csv.writer(file)
            writer.writerow(angles_to_log)

    def position_to_csv(self, position: Point, digits: int = 2) -> None:
        position_to_log: list[float] = [
            round(value, digits) for value in position]
        with open(self.position_filename, 'a+') as file:
            writer = csv.writer(file)
            writer.writerow(position_to_log)

    def target_to_csv(self, target: Point, digits: int = 2) -> None:
        target_to_log: list[float] = [
            round(value, digits) for value in target]
        with open(self.target_filename, 'a+') as file:
            writer = csv.writer(file)
            writer.writerow(target_to_log)
