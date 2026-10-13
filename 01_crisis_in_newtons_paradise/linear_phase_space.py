"""
The Chaos Toolkit - Act 1: Linear State-Space & Jacobian Spectra
"""
import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp

def harmonic_oscillator_derivatives(t, state, m=1.0, k=4.0, b=0.4):
    x1, x2 = state
    dx1 = x2
    dx2 = -(k / m) * x1 - (b / m) * x2
    return [dx1, dx2]

def compute_spectral_properties(m=1.0, k=4.0, b=0.4):
    A = np.array([[0.0, 1.0], [-k / m, -b / m]])
    trace = np.trace(A)
    det = la.det(A)
    discriminant = trace**2 - 4.0 * det
    eigenvalues = la.eigvals(A)
    phase_divergence = -b / m
    if det < 0:
        morphology = "Saddle Point (Unstable)"
    elif np.isclose(trace, 0.0) and det > 0:
        morphology = "Center (Neutral Conservative)"
    elif discriminant < 0:
        morphology = "Stable Spiral Focus" if trace < 0 else "Unstable Spiral Focus"
    else:
        morphology = "Stable Node" if trace < 0 else "Unstable Node"
    return {
        "A": A, "trace": trace, "det": det, "discriminant": discriminant,
        "eigenvalues": eigenvalues, "phase_divergence": phase_divergence,
        "morphology": morphology
    }

def simulate_orbit_comparison(t_span=(0.0, 20.0), state0=(2.0, 0.0), m=1.0, k=4.0, b=0.4, n_pts=600):
    A = np.array([[0.0, 1.0], [-k / m, -b / m]])
    t_eval = np.linspace(t_span[0], t_span[1], n_pts)
    sol = solve_ivp(harmonic_oscillator_derivatives, t_span, state0, args=(m, k, b), t_eval=t_eval, rtol=1e-9, atol=1e-9)
    analytical_states = np.zeros((n_pts, 2))
    for i, t_val in enumerate(t_eval):
        analytical_states[i] = la.expm(A * t_val) @ np.array(state0)
    return t_eval, sol.y, analytical_states.T
