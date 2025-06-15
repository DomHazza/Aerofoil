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
        self.vt = 0.0
        self.Cp = 0.0

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



# Compute tangential velocity and Cp
def compute_tangential_velocity(panels, V_inf=1.0, alpha=0.0):
    for i, pi in enumerate(panels):
        vt = V_inf * np.cos(pi.beta - alpha)
        for j, pj in enumerate(panels):
            if i != j:
                dx = pi.xc - pj.xc
                dy = pi.yc - pj.yc
                r2 = dx**2 + dy**2 + 1e-10
                vt += pj.gamma * (dx * np.cos(pi.beta) + dy * np.sin(pi.beta)) / (2 * np.pi * r2)
        pi.vt = vt
        pi.Cp = 1 - (vt / V_inf)**2



# Velocity field
def get_velocity_field(x, y, panels, V_inf=1.0, alpha=0.0):
    u = V_inf * np.cos(alpha)
    v = V_inf * np.sin(alpha)
    for panel in panels:
        dx = x - panel.xc
        dy = y - panel.yc
        r2 = dx**2 + dy**2 + 1e-10
        u += -panel.gamma * dy / (2 * np.pi * r2)
        v +=  panel.gamma * dx / (2 * np.pi * r2)
    return u, v




# Main simulation
alpha_deg = 10  # angle of attack in degrees
alpha = np.radians(alpha_deg)

x, y = naca4(m=0.02, p=0.4, t=0.12)
panels = [Panel(x[i], y[i], x[i+1], y[i+1]) for i in range(len(x) - 1)]
A, RHS = build_matrix_vortex(panels, alpha=alpha)
gamma = np.linalg.solve(A, RHS)
for i, g in enumerate(gamma):
    panels[i].gamma = g
compute_tangential_velocity(panels, alpha=alpha)

# Streamline field
X, Y = np.meshgrid(np.linspace(-0.5, 1.5, 300), np.linspace(-0.5, 0.5, 200))
u, v = np.zeros_like(X), np.zeros_like(Y)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        u[i, j], v[i, j] = get_velocity_field(X[i, j], Y[i, j], panels, alpha=alpha)

# Mask near surface
mask = np.zeros_like(u, dtype=bool)
for p in panels:
    dx = X - p.xc
    dy = Y - p.yc
    dist = np.sqrt(dx**2 + dy**2)
    mask |= dist < 0.005
u = np.where(mask, np.nan, u)
v = np.where(mask, np.nan, v)



# Plot
plt.figure(figsize=(10, 4))
plt.streamplot(X, Y, u, v, density=2, linewidth=0.7, color='orange')
plt.fill(x, y, color='white', zorder=10)
plt.plot([p.xa for p in panels] + [panels[-1].xb],
         [p.ya for p in panels] + [panels[-1].yb], 'k-', lw=5)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Streamlines around airfoil (AoA={alpha_deg}°)')
plt.grid(True)
plt.tight_layout()
plt.show()
