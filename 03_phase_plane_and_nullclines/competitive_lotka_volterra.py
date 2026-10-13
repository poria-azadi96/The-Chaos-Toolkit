"""
The Chaos Toolkit - Act 3: The Phase Plane & Nullclines
Module: Strogatz Rabbits vs. Sheep Competition & Separatrix Manifold Dynamics
"""
import numpy as np
from scipy.integrate import solve_ivp

def rabbits_vs_sheep(X, Y):
    """
    Strogatz canonical competitive exclusion system:
    dx/dt = x * (3 - x - 2*y)  [Rabbits]
    dy/dt = y * (2 - x - y)    [Sheep]
    """
    dx = X * (3.0 - X - 2.0 * Y)
    dy = Y * (2.0 - X - Y)
    return dx, dy

def rabbits_vs_sheep_ode(t, state):
    x, y = state
    dx, dy = rabbits_vs_sheep(x, y)
    return [dx, dy]

def compute_rabbits_sheep_equilibria():
    eqs = [
        {"name": "Origin (Extinction)", "coord": (0.0, 0.0), "J": np.array([[3.0, 0.0], [0.0, 2.0]])},
        {"name": "Rabbit Monopoly", "coord": (3.0, 0.0), "J": np.array([[-3.0, -6.0], [0.0, -1.0]])},
        {"name": "Sheep Monopoly", "coord": (0.0, 2.0), "J": np.array([[-1.0, 0.0], [-2.0, -2.0]])},
        {"name": "Coexistence Saddle", "coord": (1.0, 1.0), "J": np.array([[-1.0, -2.0], [-1.0, -1.0]])}
    ]
    return eqs

def trace_separatrix(dt=0.01, t_span=(0.0, 10.0), epsilon=1e-3):
    J_saddle = np.array([[-1.0, -2.0], [-1.0, -1.0]])
    eigvals, eigvecs = np.linalg.eig(J_saddle)
    
    stable_idx = np.argmin(eigvals)
    v_s = eigvecs[:, stable_idx]
    v_s = v_s / np.linalg.norm(v_s)

    saddle = np.array([1.0, 1.0])
    
    init_1 = saddle + epsilon * v_s
    init_2 = saddle - epsilon * v_s

    t_eval = np.linspace(0.0, t_span[1], 400)
    
    def backwards_ode(t, state):
        dx, dy = rabbits_vs_sheep_ode(t, state)
        return [-dx, -dy]

    sol1 = solve_ivp(backwards_ode, t_span, init_1, t_eval=t_eval, rtol=1e-8, atol=1e-8)
    sol2 = solve_ivp(backwards_ode, t_span, init_2, t_eval=t_eval, rtol=1e-8, atol=1e-8)

    return sol1.y, sol2.y
