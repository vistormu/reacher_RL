import csv
import matplotlib.pyplot as plt
import bgplot as bgp

from ..logging.logging import LOGGING_PATH, PATH_FILENAME, TARGET_PATH_FILENAME, ANGLES_FILENAME


class Results:
    @staticmethod
    def show_path_plot() -> None:
        y: list[float] = []
        with open(LOGGING_PATH + PATH_FILENAME, 'r') as file:
            reader = csv.reader(file)
            for line in reader:
                y.append(float(line[1]))

        target_y: list[float] = []
        with open(LOGGING_PATH + TARGET_PATH_FILENAME, 'r') as file:
            reader = csv.reader(file)
            for line in reader:
                target_y.append(float(line[1]))

        plt.plot(y, c=bgp.Colors.black)
        plt.plot(target_y, c=bgp.Colors.red)
        plt.ioff()
        plt.show()
        plt.close()

    @staticmethod
    def show_angles_plot() -> None:
        q0: list[float] = []
        q1: list[float] = []
        q2: list[float] = []
        q3: list[float] = []
        q4: list[float] = []
        q5: list[float] = []
        with open(LOGGING_PATH + ANGLES_FILENAME, 'r') as file:
            reader = csv.reader(file)
            for line in reader:
                q0.append(float(line[0]))
                q1.append(float(line[1]))
                q2.append(float(line[2]))
                q3.append(float(line[3]))
                q4.append(float(line[4]))
                q5.append(float(line[5]))

        plt.plot(q0)
        plt.plot(q1)
        plt.plot(q2)
        plt.plot(q3)
        plt.plot(q4)
        plt.plot(q5)
        plt.ioff()
        plt.show()
        plt.close()
