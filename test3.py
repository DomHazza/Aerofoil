import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ----------------------------
# Geometry: NACA 4
# ----------------------------
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

# ----------------------------
# Vortex Panel Method
# ----------------------------
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
    A[-1, 0] = 1
    A[-1, -1] = 1
    A[-1, 1:-1] = 0
    RHS[-1] = 0
    return A, RHS

def compute_lift_coefficient(panels, V_inf=1.0, c=1.0):
    total_circulation = sum(panel.gamma * panel.length for panel in panels)
    Cl = 2 * total_circulation / (V_inf * c)
    return Cl

# ----------------------------
# Full Aero Model (Lift & Drag)
# ----------------------------
def aero_coeffs(m, p, t, alpha_deg, Re=5e6):
    x, y = naca4(m, p, t)
    panels = [Panel(x[i], y[i], x[i+1], y[i+1]) for i in range(len(x) - 1)]
    alpha = np.radians(alpha_deg)
    A, RHS = build_matrix_vortex(panels, alpha=alpha)
    gamma = np.linalg.solve(A, RHS)
    for i, g in enumerate(gamma):
        panels[i].gamma = g
    Cl = compute_lift_coefficient(panels)

    # Induced drag
    AR = 6.0
    e = 0.9
    Cd_induced = Cl**2 / (np.pi * AR * e)

    # Viscous drag (Reynolds effect modeled crudely)
    Cd_viscous = 0.0075 + 1.2*t**2 + 0.1*m
    Cd_viscous *= (1 + 1/Re**0.5)

    Cd_total = Cd_induced + Cd_viscous

    return Cl, Cd_total

# ----------------------------
# Optimization Objective
# ----------------------------
def objective(params):
    m, p, t = params
    if p < 0.01: p = 0.01
    aoa_list = [0, 5, 10]
    drag_sum = 0
    penalty = 0
    for aoa in aoa_list:
        try:
            Cl, Cd = aero_coeffs(m, p, t, aoa)
            drag_sum += Cd
            penalty += 20 * max(0, abs(Cl)-1.5)  # stall
        except:
            return 1e3
    return drag_sum + penalty

# ----------------------------
# Run Optimization
# ----------------------------
x0 = [0.02, 0.4, 0.12]
bounds = [(0.0, 0.1), (0.01, 0.9), (0.05, 0.2)]

result = minimize(objective, x0, bounds=bounds, method='Nelder-Mead', options={'maxiter': 1000})
m_opt, p_opt, t_opt = result.x

# ----------------------------
# Plot Optimized Shape
# ----------------------------
print(f"Optimized: m={m_opt:.5f}, p={p_opt:.5f}, t={t_opt:.5f}")
x, y = naca4(m_opt, p_opt, t_opt)
plt.figure(figsize=(8,3))
plt.fill(x, y, color='skyblue')
plt.plot(x, y, 'k')
plt.axis('equal')
plt.title("Optimized Airfoil")
plt.grid(True)
plt.show()

# ----------------------------
# Plot Lift Curve
# ----------------------------
aoa_range = np.linspace(-5, 15, 25)
Cl_vals, Cd_vals = [], []
for aoa in aoa_range:
    Cl, Cd = aero_coeffs(m_opt, p_opt, t_opt, aoa)
    Cl_vals.append(Cl)
    Cd_vals.append(Cd)

plt.figure(figsize=(6,4))
plt.plot(aoa_range, Cl_vals)
plt.axhline(1.5, linestyle='--', color='r', label='Stall limit')
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("Cl")
plt.title("Lift Curve")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Plot Drag Polar
# ----------------------------
plt.figure(figsize=(6,4))
plt.plot(Cd_vals, Cl_vals)
plt.xlabel("Cd")
plt.ylabel("Cl")
plt.title("Drag Polar")
plt.grid(True)
plt.show()
