from math import pi, fmod
import matplotlib.pyplot as plt
import numpy as np

def test1():
    x_axis = np.linspace(-pi, pi)
    y_axis = np.zeros_like(x_axis)
    target = 1/2 * pi
    for i in range(len(x_axis)):
        y_axis[i] = fmod((x_axis[i] - target) + pi, 2*pi) - pi

    plt.plot(x_axis, y_axis)
    plt.xlabel("angle")
    plt.ylabel("error")
    plt.show()
    
def test2():
    x_axis = np.linspace(-2*pi, 2*pi)
    y_axis = np.zeros_like(x_axis)
    for i in range(len(x_axis)):
        y_axis[i] = fmod(x_axis[i] + 11*pi, 2*pi) - pi
        # y_axis[i] = (x_axis[i] + pi) % (2*pi) - pi

    plt.plot(x_axis, y_axis)
    plt.xlabel("angle")
    plt.ylabel("projected")
    plt.show()
test2()