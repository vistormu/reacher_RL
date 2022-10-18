import csv
from pathlib import Path

from ...core.entities import Point


class PathLogging:
    def __init__(self, logging_path: str) -> None:
        self.path_filename: str = logging_path + 'path.csv'
        self._unlink_file(self.path_filename)

    @staticmethod
    def _unlink_file(filename: str) -> None:
        if Path(filename).is_file():
            Path(filename).unlink()

    def path_to_csv(self, target: Point, digits: int = 2) -> None:
        target_to_log: list[float] = [
            round(value, digits) for value in target]
        with open(self.path_filename, 'a+') as file:
            writer = csv.writer(file)
            writer.writerow(target_to_log)
