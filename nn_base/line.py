import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.gradient import *


def f(x):
    y = 0.01 * x ** 2 + 0.1 * x
    return y

def line(f, x):
    y = f(x)

    k = numerical_diff(f, x)
    b = y - k * x

    return lambda x: k * x + b

x = np.arange(0.0, 20.0, 0.1)
y = f(x)
f_line = line(f, x=5)
y_line = f_line(x)

plt.plot(x, y)
plt.plot(x, y_line)
plt.show()