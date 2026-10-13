"""
The Chaos Toolkit - Act 2: 1D Flows & Bifurcations
Module: Normal Forms of Local Bifurcations & Bottleneck Scaling
"""
import numpy as np

def saddle_node_flow(x, mu):
    """Normal form of Saddle-Node (Fold) Bifurcation: dx/dt = mu - x^2"""
    return mu - (x ** 2)

def transcritical_flow(x, mu):
    """Normal form of Transcritical Bifurcation: dx/dt = mu*x - x^2"""
    return (mu * x) - (x ** 2)

def pitchfork_supercritical_flow(x, mu):
    """Normal form of Supercritical Pitchfork: dx/dt = mu*x - x^3"""
    return (mu * x) - (x ** 3)

def pitchfork_subcritical_flow(x, mu):
    """Normal form of Subcritical Pitchfork with quintic saturation: dx/dt = mu*x + x^3 - x^5"""
    return (mu * x) + (x ** 3) - (x ** 5)

def measure_saddle_node_bottleneck(mu, x_init=-2.0, x_target=2.0, dt=0.001, max_t=500.0):
    """
    Simulates passage time through the saddle-node 'ghost' bottleneck for mu > 0.
    Theory predicts escape time T ~ pi / sqrt(mu) as mu -> 0+.
    """
    if mu <= 0:
        return np.nan
    t = 0.0
    x = x_init
    while x < x_target and t < max_t:
        dx = saddle_node_flow(x, mu)
        x += dx * dt
        t += dt
    return t
