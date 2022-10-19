import csv
import matplotlib.pyplot as plt
import bgplot as bgp


class Results:
    @staticmethod
    def show_path_plot(positions_path: str, path_path: str) -> None:
        x: list[float] = []
        y: list[float] = []
        z: list[float] = []
        with open(positions_path, 'r') as file:
            reader = csv.reader(file)
            for line in reader:
                x_str: str = line[-1].split(',')[0].split('[')[1].strip()
                y_str: str = line[-1].split(',')[1].strip()
                z_str: str = line[-1].split(',')[2].split(']')[0].strip()

                x.append(float(x_str))
                y.append(float(y_str))
                z.append(float(z_str))

        target_x: list[float] = []
        target_y: list[float] = []
        target_z: list[float] = []
        with open(path_path, 'r') as file:
            reader = csv.reader(file)
            for line in reader:
                target_x.append(float(line[0]))
                target_y.append(float(line[1]))
                target_z.append(float(line[2]))

        plt.subplot(3, 1, 1)
        plt.plot(x, c=bgp.Colors.red)
        plt.plot(target_x, c=bgp.Colors.black)

        plt.subplot(3, 1, 2)
        plt.plot(y, c=bgp.Colors.green)
        plt.plot(target_y, c=bgp.Colors.black)

        plt.subplot(3, 1, 3)
        plt.plot(z, c=bgp.Colors.blue)
        plt.plot(target_z, c=bgp.Colors.black)

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
