"""
The Chaos Toolkit - Act 4: Limit Cycles & Discrete Chaos
Module: Logistic Map, Cobweb Dynamics, Feigenbaum Cascade & Lyapunov Spectrum
"""
import numpy as np

def logistic_map(x, r):
    """Canonical 1D logistic map: x_{n+1} = r * x_n * (1 - x_n)"""
    return r * x * (1.0 - x)

def generate_bifurcation_diagram(r_min=2.5, r_max=4.0, n_r=1000, n_transient=400, n_sample=150):
    """
    Generates high-resolution Feigenbaum bifurcation diagram coordinates.
    Filters transient iterations to capture only the invariant asymptotic attractor.
    """
    r_values = np.linspace(r_min, r_max, n_r)
    r_out = []
    x_out = []

    for r in r_values:
        x = 0.5
        for _ in range(n_transient):
            x = r * x * (1.0 - x)
        
        orbit = np.zeros(n_sample)
        for i in range(n_sample):
            x = r * x * (1.0 - x)
            orbit[i] = x
        
        r_out.extend([r] * n_sample)
        x_out.extend(orbit)

    return np.array(r_out), np.array(x_out)

def compute_lyapunov_spectrum(r_min=2.5, r_max=4.0, n_r=800, n_transient=300, n_iter=600):
    """
    Computes 1D discrete Lyapunov exponent spectrum:
    lambda(r) = lim_{N -> inf} 1/N * sum_{n=0}^{N-1} ln |f'(x_n)|
    where f'(x) = r * (1 - 2*x).
    """
    r_values = np.linspace(r_min, r_max, n_r)
    lyapunov_spectrum = np.zeros(n_r)

    for idx, r in enumerate(r_values):
        x = 0.5
        for _ in range(n_transient):
            x = r * x * (1.0 - x)
        
        sum_deriv = 0.0
        for _ in range(n_iter):
            deriv = abs(r * (1.0 - 2.0 * x))
            sum_deriv += np.log(max(deriv, 1e-12))
            x = r * x * (1.0 - x)
        
        lyapunov_spectrum[idx] = sum_deriv / n_iter

    return r_values, lyapunov_spectrum

def compute_cobweb_points(r=3.2, x0=0.2, n_iter=50):
    """
    Generates staircase coordinates (x, y) for cobweb plot stepper:
    (x0, 0) -> (x0, x1) -> (x1, x1) -> (x1, x2) -> (x2, x2) -> ...
    """
    x_path = [x0]
    y_path = [0.0]
    curr_x = x0

    for _ in range(n_iter):
        next_x = logistic_map(curr_x, r)
        # Vertical segment to f(x)
        x_path.append(curr_x)
        y_path.append(next_x)
        # Horizontal segment to identity line y = x
        x_path.append(next_x)
        y_path.append(next_x)
        curr_x = next_x

    return np.array(x_path), np.array(y_path)
