from pathlib import Path

from ..core.entities import Point
from ..manipulator.entities import ManipulatorData

from .use_cases import ManipulatorLogging, PathLogging, Results

LOGGING_PATH = 'src/features/logging/results/'
MANIPULATOR_LOGGING_PATH: str = LOGGING_PATH + 'manipulator/'
PATH_LOGGING_PATH: str = LOGGING_PATH + 'path/'


class Logging:
    def __init__(self) -> None:
        # Paths
        Path(MANIPULATOR_LOGGING_PATH).mkdir(parents=True, exist_ok=True)
        Path(PATH_LOGGING_PATH).mkdir(parents=True, exist_ok=True)

        # Use cases
        self.manipulator_logging: ManipulatorLogging = ManipulatorLogging(MANIPULATOR_LOGGING_PATH)
        self.path_logging: PathLogging = PathLogging(PATH_LOGGING_PATH)

    def log_manipulator_data(self, manipulator_data: ManipulatorData) -> None:
        self.manipulator_logging.angles_to_csv(manipulator_data.angles)
        self.manipulator_logging.positions_to_csv(manipulator_data.positions)

    def log_path(self, target: Point) -> None:
        self.path_logging.path_to_csv(target)

    def show_results(self):
        Results.show_path_plot(self.manipulator_logging.positions_filename,
                               self.path_logging.path_filename)
        Results.show_angles_plot(self.manipulator_logging.angles_filename)
