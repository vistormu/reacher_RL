import csv
import matplotlib.pyplot as plt
import bgplot as bgp

from ....core import Logger


class Results:
    @staticmethod
    def show_path_plot(positions_path: str, path_path: str) -> None:
        y: list[float] = []
        with open(positions_path, 'r') as file:
            reader = csv.reader(file)
            for line in reader:
                y.append(float(line[-1][6:10].split(',')[0]))

        target_y: list[float] = []
        with open(path_path, 'r') as file:
            reader = csv.reader(file)
            for line in reader:
                target_y.append(float(line[1]))

        plt.plot(y, c=bgp.Colors.black)
        plt.plot(target_y, c=bgp.Colors.red)
        plt.ioff()
        plt.show()
        plt.close()

    @staticmethod
    def show_angles_plot(angles_path: str) -> None:
        q0: list[float] = []
        q1: list[float] = []
        q2: list[float] = []
        q3: list[float] = []
        q4: list[float] = []
        q5: list[float] = []
        with open(angles_path, 'r') as file:
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
