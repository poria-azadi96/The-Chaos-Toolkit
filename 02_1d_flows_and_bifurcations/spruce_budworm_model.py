"""
The Chaos Toolkit - Act 2: 1D Flows & Bifurcations
Module: Ludwig-Jones-Holling Spruce Budworm Outbreak & Hysteresis Engine
"""
import numpy as np

def budworm_growth(x, r, k):
    """Logistic growth component: r * x * (1 - x / k)"""
    return r * x * (1.0 - (x / k))

def bird_predation(x):
    """Holling Type-III sigmoidal predation: x^2 / (1 + x^2)"""
    return (x ** 2) / (1.0 + (x ** 2))

def budworm_derivatives(t, x, r=0.45, k=15.0):
    """Complete dimension-free Spruce Budworm ODE: dx/dt = r*x*(1 - x/k) - x^2/(1 + x^2)"""
    return budworm_growth(x, r, k) - bird_predation(x)

def simulate_budworm_hysteresis(r=0.45, k_values=None, dt=0.05, steps_per_k=400):
    """
    Drives carrying capacity k forward (maturing forest) and backward (pesticide/die-off)
    to trace the catastrophic forward outbreak and sluggish backward recovery (Hysteresis loop).
    """
    if k_values is None:
        k_values = np.linspace(5.0, 25.0, 100)
    
    forward_x = []
    x_state = 0.5  # Start in low-density refuge
    for k in k_values:
        for _ in range(steps_per_k):
            dx = budworm_derivatives(0, x_state, r=r, k=k)
            x_state = max(0.0, x_state + dx * dt)
        forward_x.append(x_state)

    backward_x = []
    for k in reversed(k_values):
        for _ in range(steps_per_k):
            dx = budworm_derivatives(0, x_state, r=r, k=k)
            x_state = max(0.0, x_state + dx * dt)
        backward_x.append(x_state)
    backward_x.reverse()

    return k_values, np.array(forward_x), np.array(backward_x)
