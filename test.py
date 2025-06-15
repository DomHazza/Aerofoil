from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np




def naca4(m, p, t, c=1.0, n=100):
    x = (1 - np.cos(np.linspace(0, np.pi, n))) * c / 2  # Cosine spacing
    yt = 5*t*c*(0.2969*np.sqrt(x/c) - 0.1260*(x/c) - 0.3516*(x/c)**2 + 0.2843*(x/c)**3 - 0.1015*(x/c)**4)
    yc = np.where(x < p*c, 
                  m*(x/(p**2))*(2*p - x/c), 
                  m*((c - x)/((1 - p)**2))*(1 + x/c - 2*p))
    dyc_dx = np.where(x < p*c, 
                      2*m/(p**2)*(p - x/c), 
                      2*m/((1 - p)**2)*(p - x/c))
    theta = np.arctan(dyc_dx)
    
    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)
    
    x_points = np.concatenate([xu[::-1], xl[1:]])
    y_points = np.concatenate([yu[::-1], yl[1:]])
    return x_points, y_points



class Panel:
    def __init__(self, xa, ya, xb, yb):
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb
        self.xc = 0.5 * (xa + xb)
        self.yc = 0.5 * (ya + yb)
        self.length = np.hypot(xb - xa, yb - ya)
        
        dx, dy = xb - xa, yb - ya
        self.beta = np.arctan2(dy, dx)  # Panel orientation
        self.sine = np.sin(self.beta)
        self.cosine = np.cos(self.beta)
        
        self.sigma = 0.0  # source strength
        self.vt = 0.0     # tangential velocity
        self.Cp = 0.0     # pressure coefficient


def build_matrix(panels):
    N = len(panels)
    A = np.zeros((N, N))
    RHS = np.zeros(N)
    
    for i, pi in enumerate(panels):
        RHS[i] = -np.cos(pi.beta)  # freestream in x-direction
        
        for j, pj in enumerate(panels):
            if i == j:
                A[i, j] = 0.5
            else:
                dx = pi.xc - pj.xa
                dy = pi.yc - pj.ya
                r = np.hypot(dx, dy)
                A[i, j] = (1 / (2 * np.pi)) * np.log(r)
    return A, RHS



def compute_tangential_velocity(panels, V_inf=1.0):
    N = len(panels)
    for i, pi in enumerate(panels):
        vt = V_inf * np.sin(pi.beta)  # Freestream contribution
        for j, pj in enumerate(panels):
            if i != j:
                dx = pi.xc - pj.xa
                dy = pi.yc - pj.ya
                r = np.hypot(dx, dy)
                vt += -pj.sigma / (2 * np.pi * r)
        pi.vt = vt
        pi.Cp = 1 - (vt / V_inf)**2



def get_velocity_at_point(x, y, panels, V_inf=1.0, alpha=0.0):
    u = V_inf * np.cos(alpha)
    v = V_inf * np.sin(alpha)
    
    for panel in panels:
        dx = x - panel.xc
        dy = y - panel.yc
        r_squared = dx**2 + dy**2 + 1e-10
        u += panel.sigma / (2 * np.pi) * dx / r_squared
        v += panel.sigma / (2 * np.pi) * dy / r_squared
    return u, v


X, Y = np.meshgrid(np.linspace(-0.5, 1.5, 150), np.linspace(-0.5, 0.5, 150))
u_field = np.zeros_like(X)
v_field = np.zeros_like(Y)

# Compute velocity at each grid point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        u_field[i, j], v_field[i, j] = get_velocity_at_point(X[i, j], Y[i, j], panels)



x, y = naca4(m=0.02, p=0.4, t=0.12, c=1.0, n=100)
panels = [Panel(x[i], y[i], x[i+1], y[i+1]) for i in range(len(x) - 1)]
A, RHS = build_matrix(panels)
sigma = np.linalg.solve(A, RHS)

for i, panel in enumerate(panels):
    panel.sigma = sigma[i]



airfoil_path = Path([(p.xa, p.ya) for p in panels] + [(panels[-1].xb, panels[-1].yb)])
mask = airfoil_path.contains_points(np.c_[X.ravel(), Y.ravel()])
mask = mask.reshape(X.shape)
u_field = np.where(mask, np.nan, u_field)
v_field = np.where(mask, np.nan, v_field)

xc = [p.xc for p in panels]
Cp = [p.Cp for p in panels]

plt.figure(figsize=(10, 4))
plt.streamplot(X, Y, u_field, v_field, density=2, linewidth=0.7, arrowsize=1)

# Plot aerofoil surface
x_surface = [p.xa for p in panels] + [panels[-1].xb]
y_surface = [p.ya for p in panels] + [panels[-1].yb]
plt.plot(x_surface, y_surface, color='k', lw=2)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Streamlines around 2D aerofoil (panel method)')
plt.axis('equal')
plt.grid(True)
plt.show()