"""
The Chaos Toolkit - Act 5: Lorenz & 3D Chaos
Module: 3D Lorenz Vector Field, Volume Contraction & Butterfly Sensitive Divergence
"""
import numpy as np
from scipy.integrate import solve_ivp

def lorenz_derivatives(t, state, sigma=10.0, r=28.0, b=8.0/3.0):
    """
    Canonical 3D Lorenz system (1963):
    dx/dt = sigma * (y - x)
    dy/dt = r * x - y - x * z
    dz/dt = x * y - b * z
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = r * x - y - x * z
    dz = x * y - b * z
    return [dx, dy, dz]

def compute_lorenz_equilibria(sigma=10.0, r=28.0, b=8.0/3.0):
    """
    Computes analytical fixed points of the Lorenz system:
    - Origin (0, 0, 0)
    - C+ and C- for r > 1: (+/- sqrt(b*(r-1)), +/- sqrt(b*(r-1)), r-1)
    """
    eqs = [{"name": "Origin", "coord": (0.0, 0.0, 0.0)}]
    if r > 1.0:
        c_x = np.sqrt(b * (r - 1.0))
        c_z = r - 1.0
        eqs.append({"name": "C+", "coord": (c_x, c_x, c_z)})
        eqs.append({"name": "C-", "coord": (-c_x, -c_x, c_z)})
    
    # Phase volume contraction divergence
    divergence = -(sigma + 1.0 + b)
    # Critical subcritical Hopf bifurcation threshold
    r_hopf = (sigma * (sigma + b + 3.0)) / (sigma - b - 1.0) if (sigma - b - 1.0) > 0 else np.nan

    return eqs, divergence, r_hopf

def simulate_lorenz(state0=(1.0, 1.0, 1.0), t_span=(0.0, 40.0), dt=0.01, sigma=10.0, r=28.0, b=8.0/3.0):
    """Solves the Lorenz attractor with high-precision Runge-Kutta integration."""
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        lorenz_derivatives, t_span, state0,
        args=(sigma, r, b), t_eval=t_eval, rtol=1e-9, atol=1e-9
    )
    return sol.t, sol.y[0], sol.y[1], sol.y[2]

def simulate_butterfly_divergence(state0=(1.0, 1.0, 1.0), perturbation=1e-5, t_span=(0.0, 35.0), dt=0.01, sigma=10.0, r=28.0, b=8.0/3.0):
    """
    Demonstrates Sensitive Dependence on Initial Conditions (SDIC):
    Integrates two trajectories separated initially by delta_0 and tracks Euclidean distance.
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    state_perturbed = [state0[0] + perturbation, state0[1], state0[2]]

    sol1 = solve_ivp(lorenz_derivatives, t_span, state0, args=(sigma, r, b), t_eval=t_eval, rtol=1e-9, atol=1e-9)
    sol2 = solve_ivp(lorenz_derivatives, t_span, state_perturbed, args=(sigma, r, b), t_eval=t_eval, rtol=1e-9, atol=1e-9)

    # Euclidean separation delta(t)
    delta_r = np.sqrt(
        (sol1.y[0] - sol2.y[0])**2 +
        (sol1.y[1] - sol2.y[1])**2 +
        (sol1.y[2] - sol2.y[2])**2
    )

    return sol1.t, sol1.y, sol2.y, delta_r
