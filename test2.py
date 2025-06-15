import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

# Define the NACA airfoil geometry
def naca4(m, p, t, c=1.0, n=200):
    x = (1 - np.cos(np.linspace(0, np.pi, n))) * c / 2
    yt = 5*t*c*(0.2969*np.sqrt(x/c) - 0.1260*(x/c) - 0.3516*(x/c)**2 + 0.2843*(x/c)**3 - 0.1015*(x/c)**4)
    yc = np.where(x < p*c, m*(x/(p**2))*(2*p - x/c), m*((c - x)/((1 - p)**2))*(1 + x/c - 2*p))
    dyc_dx = np.where(x < p*c, 2*m/(p**2)*(p - x/c), 2*m/((1 - p)**2)*(p - x/c))
    theta = np.arctan(dyc_dx)
    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)
    x_points = np.concatenate([xu[::-1], xl[1:]])
    y_points = np.concatenate([yu[::-1], yl[1:]])
    return x_points, y_points

# Panel class
class Panel:
    def __init__(self, xa, ya, xb, yb):
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb
        self.xc = 0.5 * (xa + xb)
        self.yc = 0.5 * (ya + yb)
        self.length = np.hypot(xb - xa, yb - ya)
        self.beta = np.arctan2(yb - ya, xb - xa)
        self.nx =  np.sin(self.beta)
        self.ny = -np.cos(self.beta)
        self.gamma = 0.0

# Build system for vortex panels
def build_matrix_vortex(panels, V_inf=1.0, alpha=0.0):
    N = len(panels)
    A = np.zeros((N, N))
    RHS = np.zeros(N)
    for i, pi in enumerate(panels):
        RHS[i] = -V_inf * (pi.nx * np.cos(alpha) + pi.ny * np.sin(alpha))
        for j, pj in enumerate(panels):
            dx = pi.xc - pj.xc
            dy = pi.yc - pj.yc
            r2 = dx**2 + dy**2 + 1e-10
            u = -dy / (2 * np.pi * r2)
            v =  dx / (2 * np.pi * r2)
            A[i, j] = u * pi.nx + v * pi.ny
    # Kutta condition
    A[-1, 0] = 1
    A[-1, -1] = 1
    A[-1, 1:-1] = 0
    RHS[-1] = 0
    return A, RHS

# Compute lift coefficient
def compute_lift_coefficient(panels, V_inf=1.0, c=1.0):
    total_circulation = sum(panel.gamma * panel.length for panel in panels)
    Cl = 2 * total_circulation / (V_inf * c)
    return Cl




# Main loop: sweep over angles of attack
alphas_deg = np.linspace(-10, 15, 30)
Cls = []

# Generate geometry once
x, y = naca4(m=0.02, p=0.4, t=0.12)
panels = [Panel(x[i], y[i], x[i+1], y[i+1]) for i in range(len(x) - 1)]

for alpha_deg in alphas_deg:
    alpha = np.radians(alpha_deg)
    A, RHS = build_matrix_vortex(panels, alpha=alpha)
    gamma = np.linalg.solve(A, RHS)
    for i, g in enumerate(gamma):
        panels[i].gamma = g
    Cl = compute_lift_coefficient(panels)
    Cls.append(Cl)



# Plot Cl vs AoA
plt.figure(figsize=(8,5))
plt.plot(alphas_deg, Cls, 'o-', color='navy')
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("Lift Coefficient $C_L$")
plt.title("Lift Curve for NACA 4-digit")
plt.show()