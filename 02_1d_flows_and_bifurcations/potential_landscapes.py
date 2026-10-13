"""
The Chaos Toolkit - Act 2: 1D Flows & Bifurcations
Module: Potential Landscapes, Gradient Dynamics & Energy Dissipation
"""
import numpy as np

def compute_numerical_potential(f_func, x_grid, *f_args):
    """
    Computes potential landscape V(x) = - integral f(x) dx via cumulative trapezoidal rule.
    Guarantees dV/dt = - (f(x))^2 <= 0 along physical trajectories.
    """
    velocities = f_func(x_grid, *f_args)
    dx = x_grid[1] - x_grid[0]
    potential = -np.cumsum(velocities) * dx
    potential -= np.min(potential)  # Set zero baseline
    return potential

def double_well_potential(x, mu):
    """Landau symmetric double-well potential: V(x) = -1/2 * mu * x^2 + 1/4 * x^4"""
    return -0.5 * mu * (x ** 2) + 0.25 * (x ** 4)

def subcritical_potential(x, mu):
    """Catastrophic subcritical potential: V(x) = -1/2 * mu * x^2 - 1/4 * x^4 + 1/6 * x^6"""
    return -0.5 * mu * (x ** 2) - 0.25 * (x ** 4) + (1.0 / 6.0) * (x ** 6)
