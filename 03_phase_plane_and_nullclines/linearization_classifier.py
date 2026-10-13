"""
The Chaos Toolkit - Act 3: The Phase Plane & Nullclines
Module: Jacobian Spectral Linearization & Hartman-Grobman Topological Classifier
"""
import numpy as np
import scipy.linalg as la

def classify_2d_jacobian(J):
    """
    Classifies 2D equilibrium point based on Trace, Determinant, Discriminant:
    Characteristic equation: lambda^2 - tau * lambda + Delta = 0
    """
    tau = float(np.trace(J))
    delta = float(la.det(J))
    discriminant = tau**2 - 4.0 * delta
    eigenvals = la.eigvals(J)

    if delta < 0:
        morphology = "Saddle Point (Unstable, Index 1)"
        stability = "Unstable"
    elif np.isclose(delta, 0.0):
        morphology = "Non-Isolated / Degenerate Equilibrium"
        stability = "Marginal"
    elif np.isclose(tau, 0.0) and delta > 0:
        morphology = "Center (Neutral Conservative Ellipses)"
        stability = "Lyapunov Stable"
    elif discriminant < 0:
        if tau < 0:
            morphology = "Stable Spiral Focus (Sink)"
            stability = "Asymptotically Stable"
        else:
            morphology = "Unstable Spiral Focus (Source)"
            stability = "Unstable"
    else:
        if tau < 0:
            morphology = "Stable Node (Sink)"
            stability = "Asymptotically Stable"
        else:
            morphology = "Unstable Node (Source)"
            stability = "Unstable"

    return {
        "trace": tau,
        "det": delta,
        "discriminant": discriminant,
        "eigenvalues": eigenvals,
        "morphology": morphology,
        "stability": stability
    }
