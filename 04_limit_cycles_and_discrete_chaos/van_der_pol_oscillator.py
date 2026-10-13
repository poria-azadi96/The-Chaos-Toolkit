"""
The Chaos Toolkit - Act 4: Limit Cycles & Discrete Chaos
Module: Van der Pol Oscillator, Relaxation Dynamics & Hopf Transitions
"""
import numpy as np
from scipy.integrate import solve_ivp

def van_der_pol_derivatives(t, state, mu=1.0):
    """
    Standard Lienard form of Van der Pol oscillator:
    dx/dt = y
    dy/dt = mu * (1 - x^2) * y - x
    """
    x, y = state
    dx = y
    dy = mu * (1.0 - x**2) * y - x
    return [dx, dy]

def hopf_bifurcation_derivatives(t, state, mu=0.5, omega=1.0, b=1.0):
    """
    Supercritical Hopf bifurcation normal form in Cartesian coordinates:
    dx/dt = mu * x - omega * y - b * x * (x^2 + y^2)
    dy/dt = omega * x + mu * y - b * y * (x^2 + y^2)
    """
    x, y = state
    r_sq = x**2 + y**2
    dx = mu * x - omega * y - b * x * r_sq
    dy = omega * x + mu * y - b * y * r_sq
    return [dx, dy]

def simulate_van_der_pol(mu=1.0, state0=(0.1, 0.1), t_span=(0.0, 40.0), n_pts=1000):
    """Integrates Van der Pol oscillator with high numerical precision."""
    t_eval = np.linspace(t_span[0], t_span[1], n_pts)
    sol = solve_ivp(
        van_der_pol_derivatives, t_span, state0,
        args=(mu,), t_eval=t_eval, rtol=1e-8, atol=1e-8
    )
    return sol.t, sol.y[0], sol.y[1]
