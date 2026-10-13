"""
The Chaos Toolkit - Act 3: The Phase Plane & Nullclines
Module: 2D Vector Fields, Automated Nullcline Extraction & Flow Topography
"""
import numpy as np

def compute_grid_derivatives(system_func, x_range, y_range, n_pts=25, *args):
    """
    Evaluates 2D vector field components (U, V) across a uniform meshgrid.
    Returns X, Y grids along with normalized flow direction matrices.
    """
    x_vals = np.linspace(x_range[0], x_range[1], n_pts)
    y_vals = np.linspace(y_range[0], y_range[1], n_pts)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    U, V = system_func(X, Y, *args)
    speed = np.sqrt(U**2 + V**2) + 1e-12
    U_norm = U / speed
    V_norm = V / speed
    
    return X, Y, U, V, U_norm, V_norm
