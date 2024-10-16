import matplotlib.pyplot as plt
import numpy as np




def build_u_grid(x, y):
    return None


def build_v_grid(x, y):
    return None


def build_p_grid(x, y):
    return None




class VectorGrid():
    def __init__(self) -> None:
        x_pos = np.linspace(-5, 5, 10)
        y_pos = np.linspace(-5, 5, 10)
        self.x, self.y = np.meshgrid(x_pos, y_pos)
        self.u = build_u_grid(self.x, self.y)
        self.v = build_v_grid(self.x, self.y)
        self.p = build_p_grid(self.x, self.y)