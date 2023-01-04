import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bgplot import Colors
from pathlib import Path

from ..core.entities import Point

LOGGING_PATH = 'src/features/logging/data/'
PATH_LOGGING_PATH: str = LOGGING_PATH + 'path/'
RESULTS_PATH: str = LOGGING_PATH + 'results/'


class Logging:
    def __init__(self) -> None:
        # Paths
        Path(LOGGING_PATH).mkdir(parents=True, exist_ok=True)
        Path(PATH_LOGGING_PATH).mkdir(parents=True, exist_ok=True)

    def log_path(self, end_effector_path: list[Point], angles_path: list[list[float]]) -> None:
        # End effector position
        x: list[float] = [point.x for point in end_effector_path]
        y: list[float] = [point.y for point in end_effector_path]
        z: list[float] = [point.z for point in end_effector_path]

        data: dict = {'x': x,
                      'y': y,
                      'z': z,
                      }

        # Angles
        for i, angles in enumerate(zip(*angles_path)):
            if i == 0:
                continue

            data[f'angle_{i}'] = angles  # type:ignore

        data_frame: pd.DataFrame = pd.DataFrame(data)

        data_frame.to_csv(PATH_LOGGING_PATH+'path.csv')

    def plot_results(self) -> None:
        data_frame: pd.DataFrame = pd.read_csv(PATH_LOGGING_PATH+'path.csv')

        # End effector position
        plt.figure()
        plt.grid(alpha=0.4)
        plt.title('End effector position through time')
        plt.ylabel('Position (m)')
        plt.xlabel('Time (s)')

        plt.plot(data_frame['x'].to_numpy(), c=Colors.red)
        plt.plot(data_frame['y'].to_numpy(), c=Colors.green)
        plt.plot(data_frame['z'].to_numpy(), c=Colors.blue)

        plt.legend(['x', 'y', 'z'])

        plt.savefig(RESULTS_PATH+'end_effector.png')

        # Angles
        plt.figure()
        plt.grid(alpha=0.4)
        plt.title('Manipulator joint angles through time')
        plt.ylabel('Angles (rad)')
        plt.xlabel('Time (s)')

        plt.plot(data_frame['angle_1'].to_numpy(), c=Colors.red)
        plt.plot(data_frame['angle_2'].to_numpy(), c=Colors.green)
        plt.plot(data_frame['angle_3'].to_numpy(), c=Colors.blue)
        plt.plot(data_frame['angle_4'].to_numpy(), c='#c696bc')
        plt.plot(data_frame['angle_5'].to_numpy(), c=Colors.pink)
        plt.plot(data_frame['angle_6'].to_numpy(), c='#ECC78C')

        plt.legend([r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$', r'$\theta_5$', r'$\theta_6$'])

        plt.savefig(RESULTS_PATH+'angles.png')
